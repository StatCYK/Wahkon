"""
profile.py  --  Wahkon (WKN) model
===================================
Implements the ProfileWKN model (profile-objective deep RKHS superposition
network) and all supporting infrastructure: the WKN base class, dataset
helpers, and lambda selection utilities.

WKN uses Gaussian kernel bases along every edge:
    phi(t) = sum_i c_i K(grid_i, t),   K(s,t) = exp(-(s-t)^2 / sigma^2).

ProfileWKN replaces the standard MSE + RKHS loss with the **profile
(concentrated) objective** for the last layer:

    P_last  =  n lambda_L * y^T (Q_Phi + n lambda_L I)^{-1} y

where Q_Phi = sum_k Q_k is the composite kernel built from penultimate-layer
activations.  The representer coefficient vector
alpha* = (Q_Phi + n lambda_L I)^{-1} y  is solved analytically at each step.

Two-lambda structure
--------------------
* **lamb_last** (lambda_L): last-layer profile regularisation.
* **lamb_lower** (lambda_lo): RKHS penalty for layers 0..L-2.
"""

import sys
import os
import copy
import glob

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm as _scipy_norm
from scipy.optimize import minimize as _scipy_minimize

from .core.layer import WKNLayer
from .core.spline import SS_batch


# ============================================================
#  Dataset helpers
# ============================================================

def create_dataset(f, f_true=None, n_var=2, ranges=(-1, 1),
                   train_num=1000, test_num=1000,
                   normalize_input=False, normalize_label=False,
                   device='cpu', seed=0):
    """Create a train/test dictionary for a synthetic function ``f``."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(ranges, (list, tuple)) and not isinstance(ranges[0], (list, tuple)):
        ranges = [ranges] * n_var

    train_input = torch.zeros(train_num, n_var)
    test_input  = torch.zeros(test_num,  n_var)
    for i, r in enumerate(ranges):
        train_input[:, i] = torch.FloatTensor(train_num).uniform_(r[0], r[1])
        test_input[:, i]  = torch.FloatTensor(test_num).uniform_(r[0], r[1])

    train_label = f(train_input)
    test_label  = f(test_input)
    train_true  = f_true(train_input) if f_true is not None else train_label
    test_true   = f_true(test_input)  if f_true is not None else test_label

    if normalize_input:
        mean = train_input.mean(dim=0, keepdim=True)
        std  = train_input.std(dim=0, keepdim=True).clamp(min=1e-6)
        train_input = (train_input - mean) / std
        test_input  = (test_input  - mean) / std

    if normalize_label:
        mean = train_label.mean()
        std  = train_label.std().clamp(min=1e-6)
        train_label = (train_label - mean) / std
        test_label  = (test_label  - mean) / std
        train_true  = (train_true  - mean) / std
        test_true   = (test_true   - mean) / std

    return {
        'train_input': train_input.to(device),
        'train_label': train_label.to(device),
        'test_input':  test_input.to(device),
        'test_label':  test_label.to(device),
        'train_true':  train_true.to(device),
        'test_true':   test_true.to(device),
    }


def create_dataset2(f, f_true=None, n_var=2, ranges=(-1, 1),
                    train_num=1000, test_num=1000,
                    normalize_input=False, normalize_label=False,
                    device='cpu', seed=0):
    """Alias for ``create_dataset`` (legacy name used in experiment scripts)."""
    return create_dataset(
        f, f_true=f_true, n_var=n_var, ranges=ranges,
        train_num=train_num, test_num=test_num,
        normalize_input=normalize_input, normalize_label=normalize_label,
        device=device, seed=seed,
    )


# ============================================================
#  WKN base class
# ============================================================

class WKN(nn.Module):
    """
    Wahkon (WKN) base model -- deep RKHS superposition network.

    Each layer uses Gaussian kernel bases  K(s,t) = exp(-(s-t)^2/sigma^2)
    along every edge.  ProfileWKN extends this with the profile objective.
    """

    def __init__(
        self,
        width=None,
        grid=3,
        sigma=1.0,
        noise_scale=0.1,
        noise_scale_base=1.0,
        base_fun=torch.nn.SiLU(),
        scale_base_mu=0.0,
        scale_base_sigma=None,
        bias_trainable=True,
        grid_eps=1.0,
        grid_range=(-1, 1),
        sp_trainable=True,
        sb_trainable=True,
        device='cpu',
        seed=0,
        norm_type=None,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        if width is None:
            width = [2, 5, 1]

        self.width   = list(width)
        self.depth   = len(width) - 1
        self.grid    = grid
        self.sigma   = sigma
        self.base_fun = base_fun
        self.device  = device

        if scale_base_sigma is None:
            scale_base_sigma = noise_scale_base

        self.act_fun = nn.ModuleList()
        self.biases  = nn.ModuleList()

        for l in range(self.depth):
            layer = WKNLayer(
                in_dim=width[l], out_dim=width[l + 1],
                num=grid,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=1.0,
                base_fun=base_fun,
                grid_eps=grid_eps,
                grid_range=list(grid_range),
                sp_trainable=sp_trainable,
                sb_trainable=sb_trainable,
                device=device,
                sigma=sigma,
            )
            self.act_fun.append(layer)

            bias = nn.Linear(width[l + 1], 1, bias=False, device=device)
            bias.weight.data *= 0.0
            bias.requires_grad_(bias_trainable)
            self.biases.append(bias)

        # -- BatchNorm layers (between hidden layers, not after output) ------
        self.norm_type   = norm_type
        self.norm_layers = nn.ModuleList()
        if norm_type == 'batch':
            for l in range(self.depth - 1):
                d_out = width[l + 1]
                self.norm_layers.append(nn.BatchNorm1d(d_out, device=device))
        elif norm_type is not None:
            raise ValueError(f"norm_type must be None or 'batch'; got '{norm_type}'")

    # ------------------------------------------------------------------
    def forward(self, x):
        self.acts              = [x]
        self.spline_preacts    = []
        self.spline_postsplines = []
        self.spline_postacts   = []
        self.acts_scale        = []
        self.acts_scale_std    = []

        for l in range(self.depth):
            x_num, preacts, postacts, postspline = self.act_fun[l](x)
            x = x_num

            input_range  = (self.act_fun[l].grid[:, -1] - self.act_fun[l].grid[:, 0] + 1e-4)
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range[None, :])
            self.acts_scale_std.append(torch.std(postacts, dim=0))

            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight

            if self.norm_layers and l < self.depth - 1:
                x = self.norm_layers[l](x)

            self.acts.append(x)

        return x

    # ------------------------------------------------------------------
    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    # ------------------------------------------------------------------
    def save_ckpt(self, name, folder='./model_ckpt'):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        torch.save(self.state_dict(), path)
        print(f'Saved checkpoint: {path}')

    def load_ckpt(self, name, folder='./model_ckpt'):
        path = os.path.join(folder, name)
        self.load_state_dict(torch.load(path))
        print(f'Loaded checkpoint: {path}')

    def clear_ckpts(self, folder='./model_ckpt'):
        if os.path.exists(folder):
            for f in glob.glob(os.path.join(folder, '*')):
                os.remove(f)
        else:
            os.makedirs(folder)


# ============================================================
#  ProfileWKN
# ============================================================

class ProfileWKN(WKN):
    """
    Profile-objective WKN.

    The **last layer** uses a composite RKHS kernel  Q_Phi = sum_k Q_k
    (unweighted sum) built from Gaussian kernels on the penultimate
    activations.  The representer coefficients  alpha* = (Q_Phi + n*lambda_L I)^{-1} y
    are solved at every training step, and the profile objective
    P = n*lambda_L * y^T alpha*  is back-propagated to lower-layer parameters
    via the envelope theorem.
    """

    def __init__(self, kernel_type='additive',
                 activation='none', **kwargs):
        super().__init__(**kwargs)
        self.kernel_type = kernel_type  # 'additive' or 'rbf'
        self.activation = activation    # 'none', 'sigmoid', or 'batchnorm'

        if self.activation == 'batchnorm':
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
        feats : (n, K) -- penultimate-layer activations (K = width[-2])
        sigma : float  -- kernel bandwidth
        device : str

        Returns
        -------
        Q_list : list of K tensors, each (n, n)
        """
        K = feats.shape[1]
        Q_list = []
        for k in range(K):
            x_k = feats[:, k]
            diff = x_k.unsqueeze(1) - x_k.unsqueeze(0)
            Q_k = torch.exp(-0.5 * diff ** 2 / sigma ** 2)
            Q_list.append(Q_k)
        return Q_list

    def _build_Q_cross(self, feats_test, feats_train, sigma, device):
        """Build cross-kernel matrices Q_{test,train} for prediction."""
        K = feats_train.shape[1]
        Q_cross = []
        for k in range(K):
            x_te = feats_test[:, k]
            x_tr = feats_train[:, k]
            diff = x_te.unsqueeze(1) - x_tr.unsqueeze(0)
            Q_cross.append(torch.exp(-0.5 * diff ** 2 / sigma ** 2))
        return Q_cross

    # ------------------------------------------------------------------
    #  Composite kernel builder
    # ------------------------------------------------------------------

    def _build_Q_composite(self, Q_list):
        """Build the composite kernel  Q_Phi = sum_k Q_k."""
        Q_phi = Q_list[0]
        for k in range(1, len(Q_list)):
            Q_phi = Q_phi + Q_list[k]
        return Q_phi

    def _build_Q_cross_composite(self, Q_cross_list):
        """Build the composite cross-kernel  q_Phi(X*, X_tr) = sum_k Q_k(X*, X_tr)."""
        q_phi = Q_cross_list[0]
        for k in range(1, len(Q_cross_list)):
            q_phi = q_phi + Q_cross_list[k]
        return q_phi

    # ------------------------------------------------------------------
    #  Full RBF kernel (multi-dimensional)
    # ------------------------------------------------------------------

    def _build_Q_rbf(self, feats, sigma, device):
        """Build a full RBF kernel on the multi-dimensional feature vector."""
        dist_sq = torch.cdist(feats, feats, p=2).pow(2)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    def _build_Q_cross_rbf(self, feats_test, feats_train, sigma, device):
        """Build cross-kernel for full RBF."""
        dist_sq = torch.cdist(feats_test, feats_train, p=2).pow(2)
        return torch.exp(-0.5 * dist_sq / sigma ** 2)

    # ------------------------------------------------------------------
    #  Unified Q_phi builders (dispatch on kernel_type)
    # ------------------------------------------------------------------

    def _build_Q_phi_from_feats(self, feats, sigma, device):
        """Build Q_Phi from penultimate features, dispatching on kernel_type."""
        if self.kernel_type == 'rbf':
            return self._build_Q_rbf(feats, sigma, device)
        else:
            Q_list = self._build_Q_matrices(feats, sigma, device)
            return self._build_Q_composite(Q_list)

    def _build_Q_cross_from_feats(self, feats_test, feats_train, sigma, device):
        """Build cross-kernel from features, dispatching on kernel_type."""
        if self.kernel_type == 'rbf':
            return self._build_Q_cross_rbf(feats_test, feats_train, sigma, device)
        else:
            Q_cross = self._build_Q_cross(feats_test, feats_train, sigma, device)
            return self._build_Q_cross_composite(Q_cross)

    # ------------------------------------------------------------------
    #  Last-layer subproblem solver  (profile / concentrated objective)
    # ------------------------------------------------------------------

    def _solve_last_layer(self, Q_list, y, lamb_n, device, Q_phi=None):
        """
        Compute the profile objective and representer coefficients alpha*.

        Uses the envelope theorem: alpha* is detached, gradients flow
        through Q_phi to lower-layer parameters.
        """
        if Q_phi is not None:
            n = Q_phi.shape[0]
        else:
            n = Q_list[0].shape[0]
        y_flat = y.view(-1)

        if Q_phi is None:
            Q_phi = self._build_Q_composite(Q_list)

        reg_Q = Q_phi + lamb_n * torch.eye(n, device=device)

        # Solve detached (envelope theorem)
        with torch.no_grad():
            reg_det = reg_Q.detach()
            y_det = y_flat.detach()
            try:
                alpha_star = torch.linalg.solve(reg_det, y_det)
            except Exception:
                alpha_star = torch.linalg.lstsq(reg_det, y_det).solution

        alpha_det = alpha_star.detach()

        # Differentiable surrogate (for backprop through Q_phi)
        profile_val = -lamb_n * (alpha_det @ (Q_phi @ alpha_det))

        # True profile value (for logging)
        with torch.no_grad():
            self._profile_val_true = float(lamb_n * (y_flat @ alpha_det))

        return alpha_star, profile_val

    # ------------------------------------------------------------------
    #  Lower-layer RKHS penalty (layers 0 .. L-2)
    # ------------------------------------------------------------------

    def _compute_lower_rkhs_penalty(self, device):
        """RKHS norm penalty for layers 0 to depth-2."""
        rkhs_pen = torch.tensor(0.0, device=device)
        for l in range(self.depth - 1):
            layer_pen = torch.tensor(0.0, device=device)
            in_dim_l  = self.width[l]
            out_dim_l = self.width[l + 1]
            for d_in in range(in_dim_l):
                grid_d = self.act_fun[l].grid[d_in]
                diff = grid_d.unsqueeze(1) - grid_d.unsqueeze(0)
                K_mat = torch.exp(-0.5 * diff ** 2 / self.sigma ** 2)
                for d_out in range(out_dim_l):
                    c = self.act_fun[l].coef[d_in, d_out, :]
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

            # Optional per-neuron activation
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
            Regularisation weight lambda_L for the last-layer profile objective.
        lamb_lower : float
            RKHS penalty weight for layers 0..L-2 (lower layers).
        """
        if loss_fn is None:
            loss_fn_eval = lambda pred, y: torch.mean((pred - y) ** 2)
        else:
            loss_fn_eval = loss_fn

        n_total = dataset['train_input'].shape[0]
        batch_sz = n_total if (batch == -1 or batch > n_total) else batch
        test_bsz = dataset['test_input'].shape[0]

        D_last = self.width[-2]

        # Optimiser: lower-layer parameters only
        params = []
        for l in range(self.depth - 1):
            params.extend(self.act_fun[l].parameters())
            params.extend([self.biases[l].weight])
        if self.norm_layers:
            params.extend(self.norm_layers.parameters())
        if len(params) == 0:
            params = [nn.Parameter(torch.zeros(1, device=device))]
            _depth1_dummy = True
        else:
            _depth1_dummy = False

        optimizer = torch.optim.Adam(params, lr=lr)

        results = {
            'train_loss': [], 'test_loss': [], 'test_loss2': [],
            'profile_obj': [],
        }
        if metrics:
            for m in metrics:
                results[m.__name__] = []

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

            def closure():
                nonlocal _current_train_mse, _current_profile
                optimizer.zero_grad()

                x_train = dataset['train_input'][tr_id].to(device)
                y_train = dataset['train_label'][tr_id].to(device).view(-1)
                n = x_train.shape[0]

                feats = self.forward_to_penultimate(x_train)

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

                self._last_alpha = alpha_star.detach().clone()
                self._last_feats_train = feats.detach().clone()

                lower_pen = torch.tensor(0.0, device=device)
                if lamb_lower > 0 and self.depth > 1:
                    lower_pen = self._compute_lower_rkhs_penalty(device)
                    lower_pen = torch.clamp(lower_pen, max=1e8)

                objective = profile_val + lamb_lower * lower_pen

                if torch.isfinite(objective):
                    objective.backward()

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
        Predict at ``x_test`` using the stored representer coefficients alpha*.

        Must call ``train()`` or ``fit_last_layer()`` first.
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
        """Estimate the observation noise variance from training residuals."""
        with torch.no_grad():
            feats = self.forward_to_penultimate(
                dataset['train_input'].to(device)
            )
            n = feats.shape[0]
            y = dataset['train_label'].to(device).view(-1)

            K_tt = self._build_Q_phi_from_feats(feats, self.sigma, device)

            lamb_n = lamb_last * n
            reg_K = K_tt + lamb_n * torch.eye(n, device=device)

            try:
                L = torch.linalg.cholesky(reg_K)
                S = torch.cholesky_solve(K_tt, L)
            except Exception:
                S = torch.linalg.lstsq(reg_K, K_tt).solution

            f_hat = K_tt @ torch.linalg.solve(reg_K, y)
            residuals = y - f_hat
            rss = float(torch.sum(residuals ** 2))

            tr_S = float(torch.trace(S))
            dof = max(n - tr_S, 1.0)

            sigma2_simple = rss / n
            sigma2_dof = rss / dof

        return sigma2_simple, sigma2_dof

    def inference(self, dataset, lamb_last, noise_var=None, device='cpu'):
        """
        Posterior mean and confidence intervals at test points.

        Returns
        -------
        pred_mean : (n_test,)
        pred_cov  : None
        pred_std  : (n_test,)
        info      : dict
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

        D_last = self.width[-2]
        K_tt = self._build_Q_phi_from_feats(feats_train, self.sigma, device)
        K_pt = self._build_Q_cross_from_feats(
            feats_test, feats_train, self.sigma, device,
        )

        if self.kernel_type == 'rbf':
            K_pp_diag = torch.ones(n_test, device=device)
        else:
            K_pp_diag = torch.full((n_test,), float(D_last), device=device)

        lamb_n = lamb_last * n_train
        reg_K = K_tt + lamb_n * torch.eye(n_train, device=device)

        try:
            L = torch.linalg.cholesky(reg_K)
            A_train = torch.cholesky_solve(K_tt, L)
            alpha = torch.cholesky_solve(y_train.unsqueeze(1), L).squeeze(1)
            V = torch.linalg.solve_triangular(L, K_pt.T, upper=False)
        except Exception:
            A_train = torch.linalg.lstsq(reg_K, K_tt).solution
            alpha = torch.linalg.lstsq(reg_K, y_train).solution
            V = None

        pred_mean = K_pt @ alpha

        f_hat_train = A_train @ y_train
        residuals = y_train - f_hat_train
        tr_A = float(torch.trace(A_train))
        if noise_var is None:
            rss = float(torch.sum(residuals ** 2))
            dof = max(n_train - tr_A, 1.0)
            sigma2 = rss / dof
        else:
            sigma2 = float(noise_var)

        if V is not None:
            quad_diag = torch.sum(V ** 2, dim=0)
        else:
            M = torch.linalg.lstsq(reg_K, K_pt.T).solution
            quad_diag = torch.sum(K_pt.T * M, dim=0)

        gp_var = K_pp_diag - quad_diag
        post_var = (sigma2 / lamb_n) * gp_var
        pred_std = torch.sqrt(torch.clamp(post_var, min=0.0))

        info = {
            'noise_var': sigma2,
            'lamb_n': lamb_n,
            'tr_A': tr_A,
            'eff_dof': n_train - tr_A,
        }

        return pred_mean, None, pred_std, info

    profile_inference = inference


# ============================================================
#  Utility: lambda scaling formula
# ============================================================

def num_link_fun(width):
    """Total number of univariate link functions in the network."""
    return sum(width[l] * width[l + 1] for l in range(len(width) - 1))


def lamb_scale(n_train, width):
    """Default lower-layer penalty scaling: n^{-4/5} × #links."""
    return n_train ** (-4 / 5) * num_link_fun(width)


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
    function builds Q_Phi from the current penultimate features and evaluates
    the GCV criterion over a grid of candidate lambda values.
    """
    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1)
    n = x_train.shape[0]

    with torch.no_grad():
        feats = model.forward_to_penultimate(x_train)
        Q_phi = model._build_Q_phi_from_feats(feats, model.sigma, device)

    d, U = torch.linalg.eigh(Q_phi)
    d = torch.clamp(d, min=0.0)
    z = U.T @ y_train

    if lamb_candidates is None:
        lamb_candidates = np.logspace(
            np.log10(lamb_range[0]), np.log10(lamb_range[1]), n_candidates,
        )

    gcv_scores = {}
    for lb in lamb_candidates:
        nlamb = n * lb
        denom = d + nlamb
        resid_sq = torch.sum((nlamb * z / denom) ** 2)
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
    """
    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1)
    n = x_train.shape[0]

    with torch.no_grad():
        feats = model.forward_to_penultimate(x_train)
        Q_phi = model._build_Q_phi_from_feats(feats, model.sigma, device)

    d, U = torch.linalg.eigh(Q_phi)
    d = torch.clamp(d, min=1e-12)
    z = U.T @ y_train

    if lamb_candidates is None:
        lamb_candidates = np.logspace(
            np.log10(lamb_range[0]), np.log10(lamb_range[1]), n_candidates,
        )

    log_2pi = float(np.log(2.0 * np.pi))

    mll_scores = {}
    for lb in lamb_candidates:
        nlamb = n * lb
        denom = d + nlamb
        data_fit = float(torch.sum(z ** 2 / denom))
        log_det = float(torch.sum(torch.log(denom)))
        mll = -0.5 * (data_fit + log_det + n * log_2pi)
        mll_scores[float(lb)] = mll

    best_lamb = max(mll_scores, key=mll_scores.get)
    return best_lamb, mll_scores


# ============================================================
#  Two-stage lambda selection
# ============================================================

def select_lambda_twostage(
    width,
    dataset,
    n_splits=5,
    steps=500,
    lr=0.005,
    grid=9,
    sigma=1.0,
    n_calls=15,
    n_random_init=5,
    lamb_last_range=(0.01, 3.0),
    batch=200,
    xi=0.01,
    device='cpu',
    random_state=42,
    seed=42,
    model_kwargs=None,
):
    """
    Two-stage hyperparameter selection (faithful to paper methodology):

    1. **lamb_lower is fixed** by the deterministic formula:
       ``lamb_lower = n^{-4/5} × num_link_fun(width)``

    2. **lamb_last is selected via 1D Bayesian optimisation** over K-fold
       cross-validation RMSE.  Each BO trial trains a fresh ProfileWKN
       from scratch with the candidate ``lamb_last`` and the fixed
       ``lamb_lower``, then scores by validation RMSE.

    Parameters
    ----------
    width : list of int
        Network width specification.
    dataset : dict
        Dataset dictionary from ``create_dataset``.
    n_splits : int
        Number of CV folds (default 5).
    steps : int
        Training steps per fold (default 500).
    lr : float
        Learning rate (default 0.005).
    grid : int
        Number of kernel grid points (default 9).
    sigma : float
        Gaussian kernel bandwidth (default 1.0).
    n_calls : int
        Total BO evaluations (default 15).
    n_random_init : int
        Random initialisation calls before GP surrogate (default 5).
    lamb_last_range : tuple of float
        Search range for lamb_last as *multipliers* of the scale factor.
        Default (0.01, 3.0) → actual range is [scale*0.01, scale*3.0].
    batch : int
        Mini-batch size for training (default 200).
    xi : float
        Expected Improvement exploration parameter (default 0.01).
    device : str
        Device for training (default 'cpu').
    random_state : int
        Random state for KFold and BO (default 42).
    seed : int
        Random seed for model initialisation (default 42).
    model_kwargs : dict, optional
        Extra kwargs passed to ProfileWKN constructor.

    Returns
    -------
    best_lamb_last : float
        Selected last-layer regularisation weight.
    fixed_lamb_lower : float
        Fixed lower-layer regularisation weight (= scale).
    """
    rng = np.random.RandomState(random_state)
    n_train = dataset['train_input'].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _PENALTY = 1e6

    if model_kwargs is None:
        model_kwargs = {}

    # ---- Fixed lamb_lower = n^{-4/5} × num_link_fun --------------------
    def _num_link_fun(w):
        return sum(w[l] * w[l + 1] for l in range(len(w) - 1))

    scale = n_train ** (-4 / 5) * _num_link_fun(width)
    fixed_lamb_lower = scale

    # ---- BO search range for lamb_last (absolute scale) -----------------
    lo = scale * lamb_last_range[0]
    hi = scale * lamb_last_range[1]

    # ---- BO objective: K-fold CV validation RMSE for a given lamb_last --
    def _objective(lamb_last_val):
        val_losses = []
        for train_idx, val_idx in kf.split(range(n_train)):
            fold_ds = {
                'train_input': dataset['train_input'][train_idx],
                'train_label': dataset['train_label'][train_idx],
                'test_input':  dataset['train_input'][val_idx],
                'test_label':  dataset['train_label'][val_idx],
                'test_true':   dataset['train_label'][val_idx],
            }
            try:
                mdl = ProfileWKN(
                    width=width, grid=grid, sigma=sigma,
                    seed=seed, device=device, **model_kwargs,
                )
                mdl.train(
                    fold_ds, opt='Adam', lr=lr, steps=steps,
                    lamb_last=lamb_last_val, lamb_lower=fixed_lamb_lower,
                    batch=batch, update_grid=False,
                    verbose=False, device=device,
                    early_stopping=True, patience=50, min_delta=1e-5,
                )
                # Refit last layer with the SAME candidate lamb_last
                mdl.fit_last_layer(
                    fold_ds['train_input'], fold_ds['train_label'],
                    lamb_last=lamb_last_val, device=device,
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

    # ---- Expected Improvement acquisition function ----------------------
    def _ei(X_cand, gp, y_best):
        mu, std = gp.predict(X_cand.reshape(-1, 1), return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - xi) / std
        return (
            (y_best - mu - xi) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    # ---- Phase 1: random initialisation in LOG-SPACE --------------------
    n_random = min(n_random_init, n_calls)
    X_obs = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_random, 1)))
    y_obs = np.array([_objective(x[0]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # ---- GP surrogate (1-D Matérn-5/2) ----------------------------------
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

    # ---- Phase 2: BO loop -----------------------------------------------
    for _ in range(n_calls - n_random):
        gp.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

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
    best_lamb_last = float(X_obs[best_idx, 0])

    return best_lamb_last, fixed_lamb_lower


# ============================================================
#  2D Bayesian-optimisation lambda selection (profile objective)
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
    """Jointly select (lamb_last, lamb_lower) using 2D Bayesian optimisation."""
    rng = np.random.RandomState(random_state)
    lo_last, hi_last = lamb_last_range
    lo_lower, hi_lower = lamb_lower_range

    n_train = dataset['train_input'].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _PENALTY = 1e6

    if model_kwargs is None:
        model_kwargs = {}

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

    def _ei(X_cand, gp, y_best):
        mu, std = gp.predict(X_cand, return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - xi) / std
        return (
            (y_best - mu - xi) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    n_random = min(n_random_init, n_calls)
    X_obs = np.column_stack([
        rng.uniform(lo_last, hi_last, size=n_random),
        rng.uniform(lo_lower, hi_lower, size=n_random),
    ])
    y_obs = np.array([_objective(x[0], x[1]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

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

    for _ in range(n_calls - n_random):
        gp.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

        g1 = np.linspace(lo_last, hi_last, 30)
        g2 = np.linspace(lo_lower, hi_lower, 30)
        gg1, gg2 = np.meshgrid(g1, g2)
        x_grid = np.column_stack([gg1.ravel(), gg2.ravel()])
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
    """Select lambda via K-fold cross-validation on a fixed grid of candidates."""
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
