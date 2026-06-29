"""
wahkon.py  –  Profile-Objective WKN
==========================================
Replaces the standard MSE + RKHS loss in WKN with the **profile
(concentrated) objective** for the last layer.

Mathematical formulation
------------------------
Given penultimate-layer activations  x^{(L-1)}_{i,k}  for  i=1..n,  k=1..K
where K = D_{L-1} (penultimate width):

    Q_k  ∈  R^{n×n},   [Q_k]_{ij} = K(x^{(L-1)}_{i,k}, x^{(L-1)}_{j,k})

Composite kernel (unweighted sum):

    Q_Φ  =  Σ_k  Q_k

Profile objective (last-layer term):

    P_last  =  n λ_L · y^T (Q_Φ + n λ_L I)^{-1} y

The representer coefficient vector  α* = (Q_Φ + n λ_L I)^{-1} y  is solved
as an n×n system at each step.  The posterior mean at test points is:

    f̂(x*)  =  q_Φ(x*, X_tr)^T  α*

Backpropagation through Q_Φ (which depends on lower-layer parameters Φ)
is handled by the **envelope theorem**: differentiate P treating α* as
fixed (detached).

Two-lambda structure
--------------------
* **lamb_last** (λ_L): regularisation weight for the last-layer profile
  objective.  Controls how smooth the last-layer RKHS functions are.
* **lamb_lower** (λ_lo): RKHS penalty weight for layers 0..L-2.
  Controls the smoothness of the lower-layer link functions.

These are **independent** hyperparameters.  Bayesian optimisation can
search for them jointly or separately.

Hyperparameter selection
------------------------
Lambda is selected via Bayesian optimisation with a GP surrogate
(Matérn-5/2 kernel) and Expected Improvement acquisition.  Grid knots
are **fixed** (no adaptive grid updates during training).
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm as _scipy_norm
from scipy.optimize import minimize as _scipy_minimize

from wahkon_backbone import WahkonBase, create_dataset
from wahkon_backbone import LBFGS


# ============================================================
#  ProfileWKN
# ============================================================

class ProfileWKN(WahkonBase):
    """
    Profile-objective Wahkon network.

    The **last layer** uses a composite RKHS kernel  Q_Φ = Σ_k Q_k
    (unweighted sum) built from Gaussian kernels on the penultimate
    activations.  The representer coefficients  α* = (Q_Φ + nλ_L I)^{-1} y
    are solved at every training step, and the profile objective
    P = nλ_L y^T α*  is back-propagated to lower-layer Gaussian kernel
    parameters via the envelope theorem.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`WahkonBase.__init__`.
    """

    def __init__(self, kernel_type='additive',
                 activation='none', **kwargs):
        super().__init__(**kwargs)
        self.kernel_type = kernel_type  # 'additive' or 'rbf'
        self.activation = activation    # 'none', 'sigmoid', or 'batchnorm'

        # Create BatchNorm layers for previous (non-last) layers if requested
        if self.activation == 'batchnorm':
            import torch.nn as nn
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(self.width[l + 1])
                for l in range(self.depth - 1)
            ])

    # ------------------------------------------------------------------
    #  Kernel matrix construction
    # ------------------------------------------------------------------

    def _build_Q_matrices(self, feats, sigma, device):
        """
        Build Gaussian kernel matrices from penultimate-layer activations.

        Args
        ----
        feats : (n, K) – penultimate-layer activations (K = width[-2])
        sigma : float  – kernel bandwidth
        device : str

        Returns
        -------
        Q_list : list of K tensors, each (n, n)
        """
        K = feats.shape[1]
        Q_list = []
        for k in range(K):
            x_k = feats[:, k]                                    # (n,)
            diff = x_k.unsqueeze(1) - x_k.unsqueeze(0)          # (n, n)
            Q_k = torch.exp(-0.5 * diff ** 2 / sigma ** 2)
            Q_list.append(Q_k)
        return Q_list

    def _build_Q_cross(self, feats_test, feats_train, sigma, device):
        """
        Build cross-kernel matrices Q_{test,train} for prediction.

        Returns list of K matrices, each (n_test, n_train).
        """
        K = feats_train.shape[1]
        Q_cross = []
        for k in range(K):
            x_te = feats_test[:, k]                              # (n_test,)
            x_tr = feats_train[:, k]                             # (n_train,)
            diff = x_te.unsqueeze(1) - x_tr.unsqueeze(0)        # (n_test, n_train)
            Q_cross.append(torch.exp(-0.5 * diff ** 2 / sigma ** 2))
        return Q_cross

    # ------------------------------------------------------------------
    #  Composite kernel builder
    # ------------------------------------------------------------------

    def _build_Q_composite(self, Q_list):
        """
        Build the composite kernel  Q_Φ = Σ_k Q_k  (unweighted sum).

        Args
        ----
        Q_list : list of K tensors (n, n)

        Returns
        -------
        Q_phi : (n, n) tensor with autograd graph from Q_list
        """
        Q_phi = Q_list[0]
        for k in range(1, len(Q_list)):
            Q_phi = Q_phi + Q_list[k]
        return Q_phi

    def _build_Q_cross_composite(self, Q_cross_list):
        """
        Build the composite cross-kernel  q_Φ(X*, X_tr) = Σ_k Q_k(X*, X_tr).

        Args
        ----
        Q_cross_list : list of K tensors (n_test, n_train)

        Returns
        -------
        q_phi : (n_test, n_train)
        """
        q_phi = Q_cross_list[0]
        for k in range(1, len(Q_cross_list)):
            q_phi = q_phi + Q_cross_list[k]
        return q_phi

    # ------------------------------------------------------------------
    #  Full RBF kernel (multi-dimensional)
    # ------------------------------------------------------------------

    def _build_Q_rbf(self, feats, sigma, device):
        """
        Build a full RBF kernel on the multi-dimensional feature vector.

        Q(i,j) = exp(-||h_i - h_j||² / σ²)

        Args
        ----
        feats : (n, D) – penultimate-layer activations
        sigma : float  – kernel bandwidth
        device : str

        Returns
        -------
        Q_phi : (n, n) tensor
        """
        dist_sq = torch.cdist(feats, feats, p=2).pow(2)  # (n, n)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    def _build_Q_cross_rbf(self, feats_test, feats_train, sigma, device):
        """
        Build cross-kernel for full RBF.

        Q(i,j) = exp(-0.5 * ||h_test_i - h_train_j||² / σ²)

        Returns (n_test, n_train) tensor.
        """
        dist_sq = torch.cdist(feats_test, feats_train, p=2).pow(2)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    # ------------------------------------------------------------------
    #  Unified Q_phi builders (dispatch on kernel_type)
    # ------------------------------------------------------------------

    def _build_Q_phi_from_feats(self, feats, sigma, device):
        """
        Build Q_Φ from penultimate features, dispatching on kernel_type.

        For 'additive': Q_Φ = Σ_k Q_k  (unweighted sum of 1D kernels)
        For 'rbf':      Q_Φ = exp(-||h_i - h_j||² / σ²)  (full RBF)
        """
        if self.kernel_type == 'rbf':
            return self._build_Q_rbf(feats, sigma, device)
        else:
            Q_list = self._build_Q_matrices(feats, sigma, device)
            return self._build_Q_composite(Q_list)

    def _build_Q_cross_from_feats(self, feats_test, feats_train, sigma,
                                   device):
        """
        Build cross-kernel from features, dispatching on kernel_type.
        """
        if self.kernel_type == 'rbf':
            return self._build_Q_cross_rbf(feats_test, feats_train,
                                           sigma, device)
        else:
            Q_cross = self._build_Q_cross(feats_test, feats_train,
                                          sigma, device)
            return self._build_Q_cross_composite(Q_cross)

    # ------------------------------------------------------------------
    #  Last-layer subproblem solver  (profile / concentrated objective)
    # ------------------------------------------------------------------

    def _solve_last_layer(self, Q_list, y, lamb_n, device,
                           Q_phi=None):
        """
        Compute the profile objective and representer coefficients α*.

        Profile objective
        -----------------
            P = nλ_L · y^T (Q_Φ + nλ_L I)^{-1} y

        where  Q_Φ = Σ_k Q_k  (additive, unweighted) or a full RBF kernel.

        The representer coefficient vector  α* = (Q_Φ + nλ_L I)^{-1} y
        is solved as a detached n×n system (envelope theorem: treat α*
        as fixed when differentiating P w.r.t. lower-layer params).

        The profile value  P = nλ_L · y^T α*  retains gradients through
        Q_Φ (which depends on lower-layer parameters via Q_list or feats).

        Args
        ----
        Q_list  : list of K tensors (n, n), or None if Q_phi is provided
        y       : (n,) target values
        lamb_n  : float,  = n * λ_L
        device  : str
        Q_phi   : (n, n) tensor, optional – pre-built composite kernel
                  (used for 'rbf' kernel_type).  If provided, Q_list is
                  ignored.

        Returns
        -------
        alpha_star  : (n,) tensor, **detached**
        profile_val : scalar tensor with gradients through Q_phi
        """
        if Q_phi is not None:
            n = Q_phi.shape[0]
        else:
            n = Q_list[0].shape[0]
        y_flat = y.view(-1)

        # ---- composite kernel  Q_Φ  (with grad) --------------------------
        if Q_phi is None:
            Q_phi = self._build_Q_composite(Q_list)

        # ---- regularised system  (Q_Φ + nλ I) α = y -------------------
        reg_Q = Q_phi + lamb_n * torch.eye(n, device=device)

        # Solve detached (envelope theorem)
        with torch.no_grad():
            reg_det = reg_Q.detach()
            y_det = y_flat.detach()
            try:
                alpha_star = torch.linalg.solve(reg_det, y_det)
            except Exception:
                alpha_star = torch.linalg.lstsq(reg_det, y_det).solution

        # ---- profile objective (with grad through Q_phi) ---------------
        # P = nλ · y^T (Q_Φ + nλ I)^{-1} y  =  nλ · y^T α*
        #
        # Envelope theorem gradient:
        #   dP/dQ_Φ = -nλ · α* α*^T
        #
        # Since α* is detached, y^T α* is a constant w.r.t. Q_Φ (zero grad).
        # To get the correct gradient, we use a differentiable surrogate:
        #
        #   P_diff = -nλ · α*^T Q_Φ α*  +  (terms constant w.r.t. Q_Φ)
        #
        # This gives  dP_diff/dQ_Φ = -nλ · α* α*^T  (correct).
        # The constant terms ensure P_diff = P at the optimum.
        # Since y = (Q_Φ + nλI) α* at optimum:
        #   nλ · y^T α* = nλ · α*^T Q_Φ α* + n²λ² · ||α*||²
        # So:  P_diff = -nλ · α*^T Q_Φ α* + 2nλ · α*^T Q_Φ α* + n²λ²||α*||²
        #             = nλ · α*^T Q_Φ α* + n²λ²||α*||²  = nλ · y^T α*  ✓
        # But that gives +nλ α*α*^T gradient (wrong sign).
        #
        # Correct surrogate: only the gradient matters for optimisation.
        # We split P into a differentiable part and a constant:
        #   P_diff = -nλ · α*^T Q_Φ α*      [gives correct grad -nλ α*α*^T]
        #   P_true = nλ · y^T α*              [for logging, no grad needed]

        alpha_det = alpha_star.detach()

        # Differentiable part (for backprop through Q_Φ → lower layers)
        profile_val = -lamb_n * (alpha_det @ (Q_phi @ alpha_det))

        # True profile objective value (for logging only, detached)
        with torch.no_grad():
            self._profile_val_true = float(lamb_n * (y_flat @ alpha_det))

        return alpha_star, profile_val

    # ------------------------------------------------------------------
    #  Lower-layer RKHS penalty (layers 0 .. L-2)
    # ------------------------------------------------------------------

    def _compute_lower_rkhs_penalty(self, device):
        """
        RKHS norm penalty for layers 0 to depth-2.

        For each layer, computes  Σ_{d_in,d_out} cᵀ K c  normalised
        by (in_dim × out_dim) to match original convention.
        """
        rkhs_pen = torch.tensor(0.0, device=device)
        for l in range(self.depth - 1):
            layer_pen = torch.tensor(0.0, device=device)
            in_dim_l  = self.width[l]
            out_dim_l = self.width[l + 1]
            for d_in in range(in_dim_l):
                grid_d = self.act_fun[l].grid[d_in]               # (num,)
                diff = grid_d.unsqueeze(1) - grid_d.unsqueeze(0)  # (num, num)
                K_mat = torch.exp(-0.5 * diff ** 2 / self.sigma ** 2)
                for d_out in range(out_dim_l):
                    c = self.act_fun[l].coef[d_in, d_out, :]      # (num,)
                    layer_pen = layer_pen + c @ K_mat @ c
            rkhs_pen = rkhs_pen + layer_pen / (in_dim_l * out_dim_l)
        return rkhs_pen

    # ------------------------------------------------------------------
    #  Forward through lower layers only
    # ------------------------------------------------------------------

    def forward_to_penultimate(self, x):
        """
        Forward through layers 0 .. depth-2.

        Returns the penultimate-layer activations  (batch, width[-2]).
        Also populates ``self.acts`` for the lower layers (needed for
        the RKHS penalty computation).

        When depth == 1, this is a no-op and the raw input is returned.
        """
        self.acts = [x]
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []

        for l in range(self.depth - 1):
            x_num, preacts, postacts, postspline = self.act_fun[l](x)

            x = x_num

            input_range = (
                self.act_fun[l].grid[:, -1]
                - self.act_fun[l].grid[:, 0]
                + 1e-4
            )
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range[None, :])
            self.acts_scale_std.append(torch.std(postacts, dim=0))

            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight

            if self.norm_layers and l < self.depth - 1:
                x = self.norm_layers[l](x)

            # Optional per-neuron activation after each previous layer (not last)
            if l < self.depth - 1:
                if self.activation == 'sigmoid':
                    x = torch.sigmoid(x)
                elif self.activation == 'batchnorm':
                    x = self.bn_layers[l](x)

            self.acts.append(x)

        return x

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        opt='Adam',
        steps=100,
        lr=0.03,
        lamb_last=0.0,
        lamb_lower=0.0,
        update_grid=False,
        grid_update_freq=10,
        stop_grid_update_step=100,
        batch=-1,
        loss_fn=None,
        metrics=None,
        sglr_avoid=False,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder='./video',
        verbose=True,
        device='cpu',
        early_stopping=False,
        patience=20,
        min_delta=1e-5,
        **kwargs,
    ):
        """
        Train using the profile objective.

        Parameters
        ----------
        lamb_last : float
            Regularisation weight  λ_L  for the last-layer profile
            objective.  The penalty multiplier is  n × λ_L.
        lamb_lower : float
            RKHS penalty weight for layers 0..L-2 (lower layers).
            Independent of lamb_last.
        update_grid : bool
            Adaptive grid updates (default False → fixed grid).
        early_stopping : bool
            If True, stop training when test loss has not improved by
            at least `min_delta` for `patience` consecutive steps.
            Restores the best model state on exit.
        patience : int
            Number of steps to wait for improvement (default 20).
        min_delta : float
            Minimum improvement to count as progress (default 1e-5).
        """
        if loss_fn is None:
            loss_fn_eval = lambda pred, y: torch.mean((pred - y) ** 2)
        else:
            loss_fn_eval = loss_fn

        n_total = dataset['train_input'].shape[0]
        batch_sz = n_total if (batch == -1 or batch > n_total) else batch
        test_bsz = dataset['test_input'].shape[0]

        D_last = self.width[-2]   # penultimate width  K = D_{L-1}

        # ---- optimiser: lower-layer parameters only -------------------
        params = []
        for l in range(self.depth - 1):
            params.extend(self.act_fun[l].parameters())
            params.extend([self.biases[l].weight])
        if self.norm_layers:
            params.extend(self.norm_layers.parameters())
        # If depth == 1, there are no learnable lower-layer params;
        # we still create an optimiser but it will be a no-op.
        if len(params) == 0:
            params = [nn.Parameter(torch.zeros(1, device=device))]
            _depth1_dummy = True
        else:
            _depth1_dummy = False

        if opt == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        elif opt == 'LBFGS':
            optimizer = LBFGS(
                params, lr=lr, history_size=10,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-32, tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {
            'train_loss': [], 'test_loss': [], 'test_loss2': [],
            'profile_obj': [],
        }
        if metrics:
            for m in metrics:
                results[m.__name__] = []

        # Persistent storage for the last-layer solution
        self._last_alpha = None
        self._last_feats_train = None
        _current_train_mse = torch.tensor(0.0)
        _current_profile = torch.tensor(0.0)

        # Early stopping state
        _es_best_loss = float('inf')
        _es_wait = 0
        _es_best_state = None
        _es_best_step = 0

        pbar = tqdm(range(steps), desc='WKN (profile)', ncols=120,
                    disable=not verbose)

        for step in pbar:
            tr_id = np.random.choice(n_total, batch_sz, replace=False)
            te_id = np.random.choice(
                test_bsz, min(test_bsz, test_bsz), replace=False
            )

            if (update_grid and step % grid_update_freq == 0
                    and step < stop_grid_update_step):
                self.update_grid_from_samples(
                    dataset['train_input'][tr_id].to(device)
                )

            # ---- closure --------------------------------------------------
            def closure():
                nonlocal _current_train_mse, _current_profile
                optimizer.zero_grad()

                x_train = dataset['train_input'][tr_id].to(device)
                y_train = dataset['train_label'][tr_id].to(device).view(-1)
                n = x_train.shape[0]

                # Forward through lower layers
                feats = self.forward_to_penultimate(x_train)   # (n, K)

                # Build kernel matrix (dispatch on kernel_type)
                lamb_n = lamb_last * n
                if self.kernel_type == 'rbf':
                    Q_phi = self._build_Q_rbf(feats, self.sigma, device)
                    alpha_star, profile_val = self._solve_last_layer(
                        None, y_train, lamb_n, device, Q_phi=Q_phi,
                    )
                else:
                    Q_list = self._build_Q_matrices(feats, self.sigma, device)
                    alpha_star, profile_val = self._solve_last_layer(
                        Q_list, y_train, lamb_n, device,
                    )

                # Store for prediction / test evaluation
                self._last_alpha = alpha_star.detach().clone()
                self._last_feats_train = feats.detach().clone()

                # Lower-layer RKHS penalty
                lower_pen = torch.tensor(0.0, device=device)
                if lamb_lower > 0 and self.depth > 1:
                    lower_pen = self._compute_lower_rkhs_penalty(device)
                    lower_pen = torch.clamp(lower_pen, max=1e8)

                objective = profile_val + lamb_lower * lower_pen

                if torch.isfinite(objective):
                    objective.backward()

                # Train MSE for logging (no grad needed)
                with torch.no_grad():
                    if self.kernel_type == 'rbf':
                        Q_phi_det = self._build_Q_rbf(
                            feats.detach(), self.sigma, device,
                        )
                    else:
                        Q_phi_det = self._build_Q_composite(
                            [Q.detach() for Q in Q_list],
                        )
                    f_hat = Q_phi_det @ alpha_star.detach()
                    _current_train_mse = torch.mean((y_train - f_hat) ** 2)
                    _current_profile = torch.tensor(self._profile_val_true)

                return objective

            # ---- step ----------------------------------------------------
            if opt == 'LBFGS':
                optimizer.step(closure)
            else:
                closure()
                if not _depth1_dummy:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            # ---- test evaluation -----------------------------------------
            with torch.no_grad():
                x_test = dataset['test_input'][te_id].to(device)
                y_test = dataset['test_label'][te_id].to(device).view(-1)
                if self._last_alpha is not None:
                    feats_tr = self._last_feats_train
                    feats_te = self.forward_to_penultimate(x_test)

                    q_phi_cross = self._build_Q_cross_from_feats(
                        feats_te, feats_tr, self.sigma, device,
                    )
                    f_hat_test = q_phi_cross @ self._last_alpha

                    test_loss = torch.mean((y_test - f_hat_test) ** 2)
                else:
                    test_loss = torch.tensor(float('nan'))

                if 'test_true' in dataset:
                    y_true = dataset['test_true'][te_id].to(device).view(-1)
                    test_true_loss = torch.mean(
                        (y_true - f_hat_test) ** 2
                    ) if self._last_alpha is not None else test_loss
                else:
                    test_true_loss = test_loss

            results['train_loss'].append(
                float(torch.sqrt(_current_train_mse).cpu())
            )
            results['test_loss'].append(
                float(torch.sqrt(test_loss).cpu())
            )
            results['test_loss2'].append(
                float(torch.sqrt(test_true_loss).cpu())
            )
            results['profile_obj'].append(
                float(_current_profile.cpu())
            )

            if metrics:
                for m in metrics:
                    results[m.__name__].append(m().item())

            if verbose:
                pbar.set_description(
                    'train: %.2e | test: %.2e | prof: %.2e'
                    % (results['train_loss'][-1],
                       results['test_loss'][-1],
                       results['profile_obj'][-1])
                )

            # ---- early stopping check ------------------------------------
            if early_stopping:
                current_loss = results['test_loss'][-1]
                if current_loss < _es_best_loss - min_delta:
                    _es_best_loss = current_loss
                    _es_wait = 0
                    _es_best_step = step
                    import copy
                    _es_best_state = copy.deepcopy(self.state_dict())
                    _es_best_alpha = (
                        self._last_alpha.clone()
                        if self._last_alpha is not None else None
                    )
                    _es_best_feats = (
                        self._last_feats_train.clone()
                        if self._last_feats_train is not None else None
                    )
                else:
                    _es_wait += 1
                    if _es_wait >= patience:
                        if verbose:
                            print(f"\nEarly stopping at step {step} "
                                  f"(best step: {_es_best_step}, "
                                  f"best test loss: {_es_best_loss:.6f})")
                        break

        # Restore best model state if early stopping was used
        if early_stopping and _es_best_state is not None:
            self.load_state_dict(_es_best_state)
            self._last_alpha = _es_best_alpha
            self._last_feats_train = _es_best_feats

        return results, None, None

    # ------------------------------------------------------------------
    #  Refit last layer on full training data
    # ------------------------------------------------------------------

    def fit_last_layer(self, x_train, y_train, lamb_last, device='cpu'):
        """
        Re-solve the last-layer subproblem on the **full** training set.

        Call after ``train()`` to get coefficients for the complete data
        (training uses mini-batches, so the stored coefficients correspond
        to the last batch only).

        Args
        ----
        x_train   : (n, in_dim)
        y_train   : (n, 1) or (n,)
        lamb_last : float – same λ_L used in training

        Returns
        -------
        alpha_star : (n,) representer coefficients
        """
        with torch.no_grad():
            feats = self.forward_to_penultimate(x_train.to(device))
            n = x_train.shape[0]
            lamb_n = lamb_last * n
            if self.kernel_type == 'rbf':
                Q_phi = self._build_Q_rbf(feats, self.sigma, device)
                alpha_star, _ = self._solve_last_layer(
                    None, y_train.to(device).view(-1), lamb_n, device,
                    Q_phi=Q_phi,
                )
            else:
                Q_list = self._build_Q_matrices(feats, self.sigma, device)
                alpha_star, _ = self._solve_last_layer(
                    Q_list, y_train.to(device).view(-1), lamb_n, device,
                )

        self._last_alpha = alpha_star.detach().clone()
        self._last_feats_train = feats.detach().clone()
        return alpha_star

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def predict(self, x_train, x_test, device='cpu'):
        """
        Predict at ``x_test`` using the stored representer coefficients α*.

        Posterior mean:  f̂(x*) = q_Φ(x*, X_tr)^T α*

        Must call ``train()`` or ``fit_last_layer()`` first.

        Args
        ----
        x_train : (n_train, in_dim) – the same training inputs used to
                  fit the representer coefficients.
        x_test  : (n_test, in_dim)
        device  : str

        Returns
        -------
        f_hat : (n_test,)
        """
        if self._last_alpha is None:
            raise RuntimeError(
                "No representer coefficients available. "
                "Call train() or fit_last_layer() first."
            )

        with torch.no_grad():
            feats_tr = self.forward_to_penultimate(x_train.to(device))
            feats_te = self.forward_to_penultimate(x_test.to(device))
            q_phi_cross = self._build_Q_cross_from_feats(
                feats_te, feats_tr, self.sigma, device,
            )
            f_hat = q_phi_cross @ self._last_alpha.to(device)
        return f_hat

    # ------------------------------------------------------------------
    #  Bayesian inference  (profile-objective GP posterior)
    # ------------------------------------------------------------------

    def _estimate_noise_variance(self, dataset, lamb_last, device):
        """
        Estimate the observation noise variance  σ̂²  from training
        residuals using the profile-objective fitted values.

        Two estimates are returned (caller picks one):

        1.  **Simple residual variance**:
                σ̂² = ||y − f̂||² / n

        2.  **Degrees-of-freedom corrected** (GCV-style):
                σ̂² = ||y − f̂||² / (n − tr(S))
            where  S = K_Φ (K_Φ + nλ I)⁻¹  is the smoother matrix.
            This corrects for the effective number of parameters.

        Returns
        -------
        sigma2_simple : float
        sigma2_dof    : float
        """
        with torch.no_grad():
            feats = self.forward_to_penultimate(
                dataset['train_input'].to(device)
            )
            n = feats.shape[0]
            D = feats.shape[1]
            y = dataset['train_label'].to(device).view(-1)

            # Build K_Φ (train-train)
            K_tt = self._build_Q_phi_from_feats(
                feats, self.sigma, device,
            )

            lamb_n = lamb_last * n
            reg_K = K_tt + lamb_n * torch.eye(n, device=device)

            # Smoother matrix  S = K_Φ (K_Φ + nλ_L I)⁻¹
            try:
                L = torch.linalg.cholesky(reg_K)
                S = torch.cholesky_solve(K_tt, L)   # reg_K⁻¹ K_tt = S^T
                # S = K_tt @ reg_K⁻¹ ... cholesky_solve gives reg_K⁻¹ @ K_tt
                # We need S = K_tt @ reg_K⁻¹, but since reg_K and K_tt
                # are symmetric, S^T = reg_K⁻¹ K_tt, so tr(S) = tr(S^T)
            except Exception:
                S = torch.linalg.lstsq(reg_K, K_tt).solution

            f_hat = K_tt @ torch.linalg.solve(reg_K, y)
            residuals = y - f_hat
            rss = float(torch.sum(residuals ** 2))

            tr_S = float(torch.trace(S))
            dof = max(n - tr_S, 1.0)   # avoid division by zero

            sigma2_simple = rss / n
            sigma2_dof = rss / dof

        return sigma2_simple, sigma2_dof

    def inference(self, dataset, lamb_last,
                  noise_var=None, device='cpu'):
        """
        Posterior mean and confidence intervals at test points.

        CI formula
        ----------
        Hat matrix on training points:

            A = K_Φ (K_Φ + nλ_L I)⁻¹

        On training points:  μ = A y

        Noise variance estimated from residuals (dof-corrected):

            σ̂² = ||y − A y||² / (n − tr(A))

        Posterior variance at test point x*:

            Var(x*) = (σ̂² / nλ_L) [k(x*,x*) − k(x*,x_tr)(K + nλ_L I)⁻¹ k(x_tr,x*)]

        CI:  f̂(x*) ± z_{α/2} · √Var(x*)

        Parameters
        ----------
        dataset    : dict – must contain 'train_input', 'train_label',
                     'test_input'.
        lamb_last  : float – last-layer regularisation weight λ_L
        noise_var  : float or None
            Observation noise variance σ².  If None (default), estimated
            from training residuals:  σ̂² = ||y − Ay||² / n.
        device     : str

        Returns
        -------
        pred_mean : (n_test,)  – posterior mean  f̂(x*)
        pred_cov  : None       – not computed (kept for API compatibility)
        pred_std  : (n_test,)  – marginal std = √Var(x*)
        info      : dict – diagnostic info (σ̂², nλ, tr(A), etc.)
        """
        with torch.no_grad():
            feats_train = self.forward_to_penultimate(
                dataset['train_input'].to(device)
            )
            feats_test = self.forward_to_penultimate(
                dataset['test_input'].to(device)
            )

        n_train = feats_train.shape[0]
        n_test = feats_test.shape[0]
        y_train = dataset['train_label'].to(device).view(-1)

        # Build kernel matrices
        D_last = self.width[-2]   # K = D_{L-1}
        K_tt = self._build_Q_phi_from_feats(
            feats_train, self.sigma, device,
        )
        K_pt = self._build_Q_cross_from_feats(
            feats_test, feats_train, self.sigma, device,
        )
        # K_pp diagonal: self-kernel k(x*_i, x*_i)
        if self.kernel_type == 'rbf':
            # RBF: k(x, x) = exp(0) = 1.0
            K_pp_diag = torch.ones(n_test, device=device)
        else:
            # Additive: k(x, x) = Σ_k exp(0) = K  (number of components)
            K_pp_diag = torch.full(
                (n_test,), float(D_last), device=device,
            )

        # Regularised system:  (K_tt + nλ I)
        lamb_n = lamb_last * n_train
        reg_K = K_tt + lamb_n * torch.eye(n_train, device=device)

        # Solve via Cholesky
        try:
            L = torch.linalg.cholesky(reg_K)
            # Hat matrix on training points: A = K_tt (K_tt + nλI)⁻¹
            A_train = torch.cholesky_solve(K_tt, L)
            alpha = torch.cholesky_solve(y_train.unsqueeze(1), L).squeeze(1)
            # V = L⁻¹ K_pt^T,  shape (n_train, n_test)
            V = torch.linalg.solve_triangular(L, K_pt.T, upper=False)
        except Exception:
            A_train = torch.linalg.lstsq(reg_K, K_tt).solution
            alpha = torch.linalg.lstsq(reg_K, y_train).solution
            V = None

        # Posterior mean
        pred_mean = K_pt @ alpha

        # Estimate σ² from residuals with dof correction:
        #   σ̂² = ||y − Ay||² / (n − tr(A))
        f_hat_train = A_train @ y_train
        residuals = y_train - f_hat_train
        tr_A = float(torch.trace(A_train))
        if noise_var is None:
            rss = float(torch.sum(residuals ** 2))
            dof = max(n_train - tr_A, 1.0)
            sigma2 = rss / dof
        else:
            sigma2 = float(noise_var)

        # Posterior variance at test points:
        #   Var(x*) = (σ̂²/nλ) [k(x*,x*) - k(x*,x_tr)(K+nλI)⁻¹k(x_tr,x*)]
        #
        # The bracket is the GP posterior variance with "noise" = nλ:
        #   gp_var = K_pp_diag - diag(K_pt reg_K⁻¹ K_pt^T)
        #          = K_pp_diag - diag(V^T V)
        if V is not None:
            quad_diag = torch.sum(V ** 2, dim=0)  # (n_test,)
        else:
            M = torch.linalg.lstsq(reg_K, K_pt.T).solution
            quad_diag = torch.sum(K_pt.T * M, dim=0)

        gp_var = K_pp_diag - quad_diag  # (n_test,)
        post_var = (sigma2 / lamb_n) * gp_var
        pred_std = torch.sqrt(torch.clamp(post_var, min=0.0))

        info = {
            'noise_var': sigma2,
            'lamb_n': lamb_n,
            'tr_A': tr_A,
            'eff_dof': n_train - tr_A,
        }

        return pred_mean, None, pred_std, info

    # profile_inference is an alias kept for backward compatibility
    profile_inference = inference


# ============================================================
#  GCV lambda selection for the last layer
# ============================================================

def select_lambda_last_gcv(
    model,
    dataset,
    lamb_candidates=None,
    n_candidates=50,
    lamb_range=(1e-4, 10.0),
    device='cpu',
):
    """
    Select lamb_last via Generalised Cross-Validation (GCV).

    Given a **trained** ProfileWKN (lower layers already fitted), this
    function builds Q_Φ from the current penultimate features and evaluates
    the GCV criterion over a grid of candidate λ values.

    GCV criterion
    -------------
        GCV(λ) = (1/n) ||y − Q_Φ α_λ||²  /  (1 − tr(H_λ)/n)²

    where  α_λ = (Q_Φ + nλ I)^{-1} y,  H_λ = Q_Φ (Q_Φ + nλ I)^{-1}.

    Efficient computation via eigendecomposition of Q_Φ = U Λ U^T:
        tr(H_λ) = Σ_i  d_i / (d_i + nλ)
        ||y − Q_Φ α_λ||² = Σ_i  (nλ)² z_i² / (d_i + nλ)²

    where  d_i = eigenvalues,  z = U^T y.

    Parameters
    ----------
    model : ProfileWKN
        Already trained (lower layers fixed).
    dataset : dict
        Must contain 'train_input' and 'train_label'.
    lamb_candidates : array-like, optional
        Explicit list of λ values to try. If None, a log-spaced grid
        of `n_candidates` values in `lamb_range` is used.
    n_candidates : int
        Number of candidates if lamb_candidates is None.
    lamb_range : (float, float)
        Range for the log-spaced grid.
    device : str

    Returns
    -------
    best_lamb : float
        The λ that minimises GCV.
    gcv_scores : dict
        {lamb: gcv_score} for all candidates.
    """
    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1)
    n = x_train.shape[0]

    # Build Q_Φ from current penultimate features
    with torch.no_grad():
        feats = model.forward_to_penultimate(x_train)
        Q_phi = model._build_Q_phi_from_feats(
            feats, model.sigma, device,
        )

    # Eigendecomposition  Q_Φ = U diag(d) U^T
    d, U = torch.linalg.eigh(Q_phi)
    d = torch.clamp(d, min=0.0)  # numerical safety
    z = U.T @ y_train  # rotated targets, shape (n,)

    # Candidate lambdas
    if lamb_candidates is None:
        lamb_candidates = np.logspace(
            np.log10(lamb_range[0]), np.log10(lamb_range[1]), n_candidates,
        )

    gcv_scores = {}
    for lb in lamb_candidates:
        nlamb = n * lb
        denom = d + nlamb                    # (n,)
        # Residual in eigenbasis: r_i = nλ z_i / (d_i + nλ)
        resid_sq = torch.sum((nlamb * z / denom) ** 2)
        # Trace of hat matrix: tr(H) = Σ d_i / (d_i + nλ)
        tr_H = torch.sum(d / denom)
        gcv_val = float((resid_sq / n) / ((1.0 - tr_H / n) ** 2))
        gcv_scores[float(lb)] = gcv_val

    best_lamb = min(gcv_scores, key=gcv_scores.get)
    return best_lamb, gcv_scores


# ============================================================
#  Marginal log-likelihood for lamb_last selection
# ============================================================

def select_lambda_last_mll(
    model,
    dataset,
    lamb_candidates=None,
    n_candidates=50,
    lamb_range=(1e-4, 10.0),
    device='cpu',
):
    """
    Select lamb_last by maximising the GP marginal log-likelihood.

    Given a **trained** ProfileWKN (lower layers already fitted), this
    function builds Q_Φ from the current penultimate features and evaluates
    the marginal log-likelihood over a grid of candidate λ values.

    Marginal log-likelihood
    -----------------------
        log p(y | λ) = -1/2 [ yᵀ (Q_Φ + nλ I)⁻¹ y
                              + log det(Q_Φ + nλ I)
                              + n log(2π) ]

    Efficient computation via eigendecomposition of Q_Φ = U diag(d) Uᵀ:

        yᵀ (Q_Φ + nλ I)⁻¹ y  =  Σ_i  z_i² / (d_i + nλ)
        log det(Q_Φ + nλ I)   =  Σ_i  log(d_i + nλ)

    where  d_i = eigenvalues,  z = Uᵀ y.

    Parameters
    ----------
    model : ProfileWKN
        Already trained (lower layers fixed).
    dataset : dict
        Must contain 'train_input' and 'train_label'.
    lamb_candidates : array-like, optional
        Explicit list of λ values to try.  If None, a log-spaced grid
        of `n_candidates` values in `lamb_range` is used.
    n_candidates : int
        Number of candidates if lamb_candidates is None.
    lamb_range : (float, float)
        Range for the log-spaced grid.
    device : str

    Returns
    -------
    best_lamb : float
        The λ that maximises the marginal log-likelihood.
    mll_scores : dict
        {lamb: mll_value} for all candidates (higher is better).
    """
    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1)
    n = x_train.shape[0]

    # Build Q_Φ from current penultimate features
    with torch.no_grad():
        feats = model.forward_to_penultimate(x_train)
        Q_phi = model._build_Q_phi_from_feats(
            feats, model.sigma, device,
        )

    # Eigendecomposition  Q_Φ = U diag(d) Uᵀ
    d, U = torch.linalg.eigh(Q_phi)
    d = torch.clamp(d, min=1e-12)  # numerical safety
    z = U.T @ y_train  # rotated targets, shape (n,)

    # Candidate lambdas
    if lamb_candidates is None:
        lamb_candidates = np.logspace(
            np.log10(lamb_range[0]), np.log10(lamb_range[1]), n_candidates,
        )

    log_2pi = float(np.log(2.0 * np.pi))

    mll_scores = {}
    for lb in lamb_candidates:
        nlamb = n * lb
        denom = d + nlamb                              # (n,)
        # Data-fit term: yᵀ (Q_Φ + nλ I)⁻¹ y
        data_fit = float(torch.sum(z ** 2 / denom))
        # Complexity term: log det(Q_Φ + nλ I)
        log_det = float(torch.sum(torch.log(denom)))
        # Marginal log-likelihood (up to constant)
        mll = -0.5 * (data_fit + log_det + n * log_2pi)
        mll_scores[float(lb)] = mll

    best_lamb = max(mll_scores, key=mll_scores.get)
    return best_lamb, mll_scores


# ============================================================
#  Two-stage lambda selection:  BO for lamb_lower, GCV/MLL for lamb_last
# ============================================================

def select_lambda_twostage(
    width,
    dataset,
    n_splits=3,
    steps=50,
    lr=0.03,
    grid=5,
    sigma=1.0,
    n_calls=15,
    n_random_init=5,
    lamb_lower_range=(0.0, 3.0),
    lamb_last_init=0.1,
    gcv_n_candidates=50,
    gcv_lamb_range=(1e-4, 10.0),
    lamb_last_method='mll',
    xi=0.01,
    device='cpu',
    random_state=42,
    model_kwargs=None,
):
    """
    Two-stage hyperparameter selection:

    Stage 1 — 1D Bayesian optimisation for  lamb_lower  (lower-layer
    RKHS penalty).  Each CV fold trains with  lamb_last_init  as an
    initial value, then selects the optimal  lamb_last  for that fold
    via GCV or marginal log-likelihood (controlled by lamb_last_method).
    The last layer is refit with the selected  lamb_last  before
    scoring validation RMSE.

    Stage 2 — After BO picks the best lamb_lower, a final model is
    trained and the same method (GCV or MLL) selects the optimal
    lamb_last from the eigendecomposition of Q_Φ.

    Parameters
    ----------
    width, dataset, n_splits, steps, lr, grid, sigma :
        Same as select_lambda_profile_bo.
    n_calls         : int   – total 1D BO evaluations (default 15)
    n_random_init   : int   – random initial points (default 5)
    lamb_lower_range: tuple – search bounds for λ_lower
    lamb_last_init  : float – initial λ_last used during Stage 1 training
                              (refined per fold before scoring)
    gcv_n_candidates: int   – number of candidates (used in both stages)
    gcv_lamb_range  : tuple – search range for λ_last (both stages)
    lamb_last_method: str   – 'gcv' or 'mll' (default 'mll')
    xi              : float – EI exploration bonus
    device, random_state, model_kwargs : as usual

    Returns
    -------
    best_lamb_last  : float  (from GCV or MLL)
    best_lamb_lower : float  (from BO)
    """
    rng = np.random.RandomState(random_state)
    lo, hi = lamb_lower_range
    n_train = dataset['train_input'].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _PENALTY = 1e6

    if model_kwargs is None:
        model_kwargs = {}

    # Dispatcher for lamb_last selection method
    def _select_lamb_last(mdl, ds):
        if lamb_last_method == 'mll':
            return select_lambda_last_mll(
                mdl, ds,
                n_candidates=gcv_n_candidates,
                lamb_range=gcv_lamb_range,
                device=device,
            )
        else:
            return select_lambda_last_gcv(
                mdl, ds,
                n_candidates=gcv_n_candidates,
                lamb_range=gcv_lamb_range,
                device=device,
            )

    # ---- Stage 1: 1D BO for lamb_lower --------------------------------
    def _objective(lb_lower):
        val_losses = []
        for train_idx, val_idx in kf.split(range(n_train)):
            fold_train_ds = {
                'train_input': dataset['train_input'][train_idx],
                'train_label': dataset['train_label'][train_idx],
                'test_input':  dataset['train_input'][val_idx],
                'test_label':  dataset['train_label'][val_idx],
                'test_true':   dataset['train_label'][val_idx],
            }
            try:
                mdl = ProfileWKN(
                    width=width, grid=grid, sigma=sigma,
                    device=device, **model_kwargs,
                )
                mdl.train(
                    fold_train_ds, opt='Adam', lr=lr, steps=steps,
                    lamb_last=lamb_last_init, lamb_lower=lb_lower,
                    update_grid=False,
                    verbose=False, device=device,
                    early_stopping=True, patience=20, min_delta=1e-5,
                )
                # Select lamb_last on this fold's training split
                fold_ds = {
                    'train_input': dataset['train_input'][train_idx],
                    'train_label': dataset['train_label'][train_idx],
                }
                fold_lamb_last, _ = _select_lamb_last(mdl, fold_ds)
                # Refit last layer with selected lamb_last
                mdl.fit_last_layer(
                    fold_ds['train_input'], fold_ds['train_label'],
                    lamb_last=fold_lamb_last, device=device,
                )
                # Score on validation set
                with torch.no_grad():
                    f_val = mdl.predict(
                        fold_ds['train_input'],
                        dataset['train_input'][val_idx],
                        device=device,
                    )
                    y_val = dataset['train_label'][val_idx].to(device).view(-1)
                    v = float(torch.sqrt(torch.mean((f_val - y_val) ** 2)))
                val_losses.append(v if np.isfinite(v) else _PENALTY)
            except Exception:
                val_losses.append(_PENALTY)
        result = float(np.mean(val_losses))
        return result if np.isfinite(result) else _PENALTY

    # Expected Improvement (1D)
    def _ei(X_cand, gp, y_best):
        mu, std = gp.predict(X_cand.reshape(-1, 1), return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - xi) / std
        return (
            (y_best - mu - xi) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    # Phase 1: random initialisation
    n_random = min(n_random_init, n_calls)
    X_obs = rng.uniform(lo, hi, size=(n_random, 1))
    y_obs = np.array([_objective(x[0]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # GP surrogate (1D Matern-5/2)
    length_scale = 0.4 * (hi - lo + 1e-6)
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=(0.05 * (hi - lo + 1e-6), 10.0 * (hi - lo + 1e-6)),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )

    # Phase 2: BO loop
    for _ in range(n_calls - n_random):
        gp.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

        # Grid search for EI maximum
        x_grid = np.linspace(lo, hi, 200).reshape(-1, 1)
        ei_vals = _ei(x_grid.ravel(), gp, y_best)
        x_start = x_grid[np.argmax(ei_vals)]

        res = _scipy_minimize(
            fun=lambda x: -float(_ei(np.array([x[0]]), gp, y_best)[0]),
            x0=x_start,
            bounds=[(lo, hi)],
            method='L-BFGS-B',
        )
        x_next = np.clip(
            res.x if res.success else x_start.ravel(),
            lo, hi,
        ).reshape(1, 1)

        y_next = _objective(x_next[0, 0])
        y_next = y_next if np.isfinite(y_next) else _PENALTY
        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

    best_idx = int(np.argmin(y_obs))
    best_lamb_lower = float(X_obs[best_idx, 0])

    # ---- Stage 2: train final model, then select lamb_last -------------
    final_model = ProfileWKN(
        width=width, grid=grid, sigma=sigma,
        device=device, **model_kwargs,
    )
    final_model.train(
        dataset, opt='Adam', lr=lr, steps=steps,
        lamb_last=lamb_last_init, lamb_lower=best_lamb_lower,
        update_grid=False,
        verbose=False, device=device,
        early_stopping=True, patience=20, min_delta=1e-5,
    )

    best_lamb_last, _ = _select_lamb_last(final_model, dataset)

    return best_lamb_last, best_lamb_lower


# ============================================================
#  Bayesian-optimisation lambda selection (2D, profile objective)
# ============================================================

def select_lambda_profile_bo(
    width,
    dataset,
    n_splits=3,
    steps=50,
    lr=0.03,
    grid=5,
    sigma=1.0,
    n_calls=20,
    n_random_init=6,
    lamb_last_range=(0.01, 3.0),
    lamb_lower_range=(0.0, 3.0),
    xi=0.01,
    device='cpu',
    random_state=42,
    model_kwargs=None,
):
    """
    Jointly select  (λ_last, λ_lower)  using 2D Bayesian optimisation.

    Searches over both the last-layer regularisation weight  λ_last  and
    the lower-layer RKHS penalty weight  λ_lower  simultaneously.  The
    GP surrogate is 2D with an ARD Matérn-5/2 kernel.

    Parameters
    ----------
    width              : list  – network widths, e.g. [2, 5, 1]
    dataset            : dict  – from create_dataset
    n_splits           : int   – K-fold CV splits (default 3)
    steps              : int   – training steps per fold (default 50)
    lr                 : float – learning rate (default 0.03)
    grid               : int   – B-spline grid intervals (default 5)
    sigma              : float – Gaussian kernel bandwidth (default 1.0)
    n_calls            : int   – total BO evaluations (default 20)
    n_random_init      : int   – random initial points (default 6)
    lamb_last_range    : tuple – search bounds for λ_last (default (0.01, 3.0))
    lamb_lower_range   : tuple – search bounds for λ_lower (default (0.0, 3.0))
    xi                 : float – EI exploration bonus (default 0.01)
    device             : str   – 'cpu' or 'cuda'
    random_state       : int   – RNG seed
    model_kwargs       : dict  – extra kwargs for ProfileWKN constructor

    Returns
    -------
    best_lamb_last  : float
    best_lamb_lower : float
    """
    rng = np.random.RandomState(random_state)
    lo_last, hi_last = lamb_last_range
    lo_lower, hi_lower = lamb_lower_range
    bounds_2d = np.array([[lo_last, hi_last], [lo_lower, hi_lower]])

    n_train = dataset['train_input'].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _PENALTY = 1e6

    if model_kwargs is None:
        model_kwargs = {}

    # ---- objective: mean CV validation loss for (lamb_last, lamb_lower) --
    def _objective(lb_last, lb_lower):
        val_losses = []
        for train_idx, val_idx in kf.split(range(n_train)):
            cv_dataset = {
                'train_input': dataset['train_input'][train_idx],
                'train_label': dataset['train_label'][train_idx],
                'test_input':  dataset['train_input'][val_idx],
                'test_label':  dataset['train_label'][val_idx],
                'test_true':   dataset['train_label'][val_idx],
            }
            try:
                model = ProfileWKN(
                    width=width, grid=grid, sigma=sigma,
                    device=device, **model_kwargs,
                )
                results, _, _ = model.train(
                    cv_dataset, opt='Adam', lr=lr, steps=steps,
                    lamb_last=lb_last, lamb_lower=lb_lower,
                    update_grid=False,
                    verbose=False, device=device,
                    early_stopping=True, patience=20, min_delta=1e-5,
                )
                v = float(min(results['test_loss']))
                val_losses.append(v if np.isfinite(v) else _PENALTY)
            except Exception:
                val_losses.append(_PENALTY)
        result = float(np.mean(val_losses))
        return result if np.isfinite(result) else _PENALTY

    # ---- Expected Improvement ------------------------------------------
    def _ei(X_cand, gp, y_best):
        mu, std = gp.predict(X_cand, return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - xi) / std
        return (
            (y_best - mu - xi) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    # ---- Phase 1: random initialisation (Latin hypercube-ish) ----------
    n_random = min(n_random_init, n_calls)
    X_obs = np.column_stack([
        rng.uniform(lo_last, hi_last, size=n_random),
        rng.uniform(lo_lower, hi_lower, size=n_random),
    ])  # (n_random, 2)
    y_obs = np.array([
        _objective(x[0], x[1]) for x in X_obs
    ])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # ---- GP surrogate (2D, ARD Matérn-5/2) -----------------------------
    length_scales = [
        0.4 * (hi_last - lo_last),
        0.4 * (hi_lower - lo_lower + 1e-6),
    ]
    length_scale_bounds = [
        (0.05 * (hi_last - lo_last), 10.0 * (hi_last - lo_last)),
        (0.05 * (hi_lower - lo_lower + 1e-6), 10.0 * (hi_lower - lo_lower + 1e-6)),
    ]
    kernel = Matern(
        length_scale=length_scales,
        length_scale_bounds=length_scale_bounds,
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )

    # ---- Phase 2: BO loop ---------------------------------------------
    for _ in range(n_calls - n_random):
        gp.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

        # Coarse 2D grid + L-BFGS-B polish for EI maximum
        g1 = np.linspace(lo_last, hi_last, 30)
        g2 = np.linspace(lo_lower, hi_lower, 30)
        gg1, gg2 = np.meshgrid(g1, g2)
        x_grid = np.column_stack([gg1.ravel(), gg2.ravel()])  # (900, 2)
        ei_vals = _ei(x_grid, gp, y_best)
        x_start = x_grid[np.argmax(ei_vals)]

        res = _scipy_minimize(
            fun=lambda x: -float(
                _ei(np.array(x).reshape(1, 2), gp, y_best)[0]
            ),
            x0=x_start,
            bounds=[(lo_last, hi_last), (lo_lower, hi_lower)],
            method='L-BFGS-B',
        )
        x_next = np.clip(
            res.x if res.success else x_start,
            [lo_last, lo_lower],
            [hi_last, hi_lower],
        )

        y_next = _objective(x_next[0], x_next[1])
        y_next = y_next if np.isfinite(y_next) else _PENALTY
        X_obs = np.vstack([X_obs, x_next.reshape(1, 2)])
        y_obs = np.append(y_obs, y_next)

    best_idx = int(np.argmin(y_obs))
    best_lamb_last = float(X_obs[best_idx, 0])
    best_lamb_lower = float(X_obs[best_idx, 1])
    return best_lamb_last, best_lamb_lower


# ============================================================
#  Grid-search lambda selection (profile objective)
# ============================================================

def select_lambda_profile_grid(
    width,
    dataset,
    n_splits=3,
    steps=50,
    lr=0.03,
    grid=5,
    sigma=1.0,
    lamb_lower=0.0,
    candidates=None,
    device='cpu',
    model_kwargs=None,
):
    """
    Select λ via K-fold cross-validation on a fixed grid of candidates.

    Parameters
    ----------
    candidates : list of float, optional
        Lambda values to try (linear scale).
        Default: [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

    Returns
    -------
    best_lamb : float
    cv_results : dict  {lamb: mean_val_loss}
    """
    if candidates is None:
        candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

    if model_kwargs is None:
        model_kwargs = {}

    n_train = dataset['train_input'].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {}
    for lb in candidates:
        val_losses = []
        for train_idx, val_idx in kf.split(range(n_train)):
            cv_dataset = {
                'train_input': dataset['train_input'][train_idx],
                'train_label': dataset['train_label'][train_idx],
                'test_input':  dataset['train_input'][val_idx],
                'test_label':  dataset['train_label'][val_idx],
                'test_true':   dataset['train_label'][val_idx],
            }
            try:
                model = ProfileWKN(
                    width=width, grid=grid, sigma=sigma,
                    device=device, **model_kwargs,
                )
                results, _, _ = model.train(
                    cv_dataset, opt='Adam', lr=lr, steps=steps,
                    lamb_last=lb, lamb_lower=lamb_lower,
                    update_grid=False,
                    verbose=False, device=device,
                    early_stopping=True, patience=20, min_delta=1e-5,
                )
                v = float(min(results['test_loss']))
                val_losses.append(v if np.isfinite(v) else 1e6)
            except Exception:
                val_losses.append(1e6)
        cv_results[lb] = float(np.mean(val_losses))

    best = min(cv_results, key=cv_results.get)
    return best, cv_results
