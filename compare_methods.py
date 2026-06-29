#!/usr/bin/env python3
"""
compare_methods.py  --  Multi-method comparison
================================================
Compare four methods on f1--f4:

    1. Wahkon (profile)  -- profile objective + GP posterior     (RMSE + CI)
    2. MLP               -- standard neural network              (RMSE only)
    3. MLP (Deep)        -- deeper/wider MLP                     (RMSE only)
    4. BNN               -- Bayesian NN (Bayes by Backprop)      (RMSE + CI)

Metrics
-------
- Test RMSE (point prediction vs true function)   -- all 4 methods
- 95% BCI coverage / CI width                     -- methods 1, 4

All methods use the same dataset (same seed).

Usage
-----
    python compare_methods.py
"""

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.model_selection import KFold
from scipy.stats import norm as _scipy_norm
from scipy.optimize import minimize as _scipy_minimize

from wahkon import (
    ProfileWKN,
    select_lambda_last_mll,
)


# ============================================================
#  Dataset creation
# ============================================================
def create_dataset2(f, f_true=None, n_var=2, ranges=(-1, 1),
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


# ============================================================
#  Utility: Width at matched coverage
# ============================================================
def compute_width_at_coverage(residuals, pred_std, target_coverage=0.95):
    """Compute the average CI width needed to achieve a target empirical coverage.

    Given per-point residuals |y - y_hat| and predictive std, find the z-multiplier
    such that fraction `target_coverage` of points satisfy |residual| < z * pred_std.
    Then return mean(2 * z * pred_std).

    Parameters
    ----------
    residuals : array-like
        Absolute residuals |y_true - pred_mean| (1D).
    pred_std : array-like
        Per-point predictive standard deviations (1D).
    target_coverage : float
        Desired empirical coverage (default 0.95).

    Returns
    -------
    float
        Average CI width that achieves the target coverage.
        Returns NaN if pred_std contains zeros or is empty.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    pred_std = np.asarray(pred_std, dtype=np.float64)

    if len(residuals) == 0 or np.any(pred_std <= 0):
        return float('nan')

    # Normalized residuals: z_i = |residual_i| / std_i
    z_scores = residuals / pred_std

    # The z-multiplier at target coverage = quantile of z_scores
    z_target = float(np.quantile(z_scores, target_coverage))

    # Average width at this z-multiplier
    width = float(np.mean(2 * z_target * pred_std))
    return width


# ============================================================
#  Experiment configurations  (from inference.py)
# ============================================================
EXPERIMENTS = {
    'f1': {
        'f': lambda x: (torch.log(x[:, [0]] ** 2 + x[:, [1]] ** 2 + abs(torch.tan(x[:, [2]])))
                         + 1 / torch.tan(torch.pi / (1 + torch.exp(
                             x[:, [0]] ** 2 + torch.sin(6 * x[:, [1]]) + x[:, [2]] ** 2)))
                         + 0.1 * torch.randn(x.shape[0], 1)),
        'f_true': lambda x: (torch.log(x[:, [0]] ** 2 + x[:, [1]] ** 2 + abs(torch.tan(x[:, [2]])))
                               + 1 / torch.tan(torch.pi / (1 + torch.exp(
                                   x[:, [0]] ** 2 + torch.sin(6 * x[:, [1]]) + x[:, [2]] ** 2)))),
        'width': [3, 6,6, 1],
        'n_var': 3,
        'noise_std': 0.1,
    },
    'f2': {
        'f': lambda x: (
            torch.sin(torch.sum(x ** 2, dim=1, keepdim=True))
            + 0.1 * torch.randn(x.shape[0], 1)
        ),
        'f_true': lambda x: torch.sin(torch.sum(x ** 2, dim=1, keepdim=True)),
        'width': [10, 10, 10, 1],
        'n_var': 10,
        'noise_std': 0.1,
        'ranges': [-1, 1],
        'n_train_default': 200,
        'description': 'sin(x_1^2+x_2^2+...+x_10^2): smooth links, rapid oscillation in 10D',
    },
    'f3': {
        'f': lambda x: (torch.exp(0.5 * (
            torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2))
            + torch.sin(torch.pi * (x[:, [2]] ** 2 + x[:, [3]] ** 2))))
                         + 0.420433 * torch.randn(x.shape[0], 1)),
        'f_true': lambda x: torch.exp(0.5 * (
            torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2))
            + torch.sin(torch.pi * (x[:, [2]] ** 2 + x[:, [3]] ** 2)))),
        'width': [4, 4, 4, 1],
        'n_var': 4,
        'noise_std': 0.1,#0.420433/2,
    },
    'f4': {
        'f': lambda x: (
            torch.exp(torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2)))
            * torch.cos(torch.pi * x[:, [2]] * x[:, [3]])
            + 0.1 * torch.randn(x.shape[0], 1)
        ),
        'f_true': lambda x: (
            torch.exp(torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2)))
            * torch.cos(torch.pi * x[:, [2]] * x[:, [3]])
        ),
        'width': [6, 6, 6,6, 1],
        'n_var': 6,
        'noise_std': 0.1,
        'ranges': [-1, 1],
        'n_train_default': 300,
        'description': 'exp(sin(pi(x_1^2+x_2^2)))*cos(pi*x_3*x_4): deep composition + noise vars',
    },
}

# Backward-compatible aliases
EXPERIMENTS['exp5'] = EXPERIMENTS['f1']
EXPERIMENTS['sin_sum_sq'] = EXPERIMENTS['f2']
EXPERIMENTS['exp3'] = EXPERIMENTS['f3']
EXPERIMENTS['nested_comp'] = EXPERIMENTS['f4']


# ============================================================
#  Shared settings  (matched to inference.py)
# ============================================================
GRID       = 9
SIGMA      = 1.0
STEPS      = 500
LR         = 0.005
BATCH      = 200
TEST_NUM   = 1000
BO_CALLS   = 15
BO_RANDOM_INIT = 5
BO_RANGE_SSQC    = (0.01, 100.0)
BO_RANGE_PROFILE = (0.01, 3.0)
BO_CV_FOLDS = 5
BO_XI       = 0.01


def num_link_fun(width):
    return sum(width[l] * width[l + 1] for l in range(len(width) - 1))


def lamb_scale(n_train, width):
    """Lambda scaling: n^{-4/5} x num_link_fun."""
    return n_train ** (-4 / 5) * num_link_fun(width)


# ============================================================
#  Run Wahkon (profile)
# ============================================================
def _run_wahkon_core(exp_cfg, dataset, n_train, seed, device,
                            model_kwargs=None, method_name='Wahkon'):
    """Shared core for all Wahkon (profile) variants.

    Fixed lamb_lower = scale  (= n^{-4/5} x num_link_fun).
    lamb_last selected via 1-D Bayesian optimisation over K-fold CV
    validation RMSE.  Each BO evaluation trains the full model from
    scratch with the candidate lamb_last so that the lower-layer
    features are consistent with the regularisation.

    Parameters
    ----------
    model_kwargs : dict -- extra kwargs for ProfileWKN constructor
                         (kernel_type, activation, etc.)
    method_name  : str  -- label returned in the result dict
    """
    if model_kwargs is None:
        model_kwargs = {}
    width = exp_cfg['width']
    scale = lamb_scale(n_train, width)

    # Fixed lamb_lower = scale  (c = 1)
    fixed_lamb_lower = scale

    # BO search range for lamb_last (in absolute scale)
    lo = scale * BO_RANGE_PROFILE[0]
    hi = scale * BO_RANGE_PROFILE[1]

    _PENALTY = 1e6
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=BO_CV_FOLDS, shuffle=True, random_state=seed)

    # -- BO objective: K-fold CV validation RMSE for a given lamb_last --
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
                    width=width, grid=GRID, sigma=SIGMA,
                    seed=seed, device=device, **model_kwargs,
                )
                mdl.train(
                    fold_ds, opt='Adam', lr=LR, steps=STEPS,
                    lamb_last=lamb_last_val, lamb_lower=fixed_lamb_lower,
                    batch=BATCH, update_grid=False,
                    verbose=False, device=device,
                    early_stopping=True, patience=50, min_delta=1e-5,
                )
                # Refit last layer with the candidate lamb_last
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

    # -- Expected Improvement acquisition function --
    def _ei(X_cand, gp_surr, y_best):
        mu, std = gp_surr.predict(X_cand.reshape(-1, 1), return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - BO_XI) / std
        return (
            (y_best - mu - BO_XI) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    # -- Phase 1: random initialisation in log-space --
    t0 = time.time()
    n_random = min(BO_RANDOM_INIT, BO_CALLS)
    X_obs = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_random, 1)))
    y_obs = np.array([_objective(x[0]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # -- GP surrogate (1-D Matern-5/2) --
    length_scale = 0.4 * (hi - lo + 1e-6)
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=(0.05 * (hi - lo + 1e-6), 10.0 * (hi - lo + 1e-6)),
        nu=2.5,
    )
    gp_surr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )

    # -- Phase 2: BO loop --
    for _ in range(BO_CALLS - n_random):
        gp_surr.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

        # Grid search for EI maximum
        x_grid = np.linspace(lo, hi, 200).reshape(-1, 1)
        ei_vals = _ei(x_grid.ravel(), gp_surr, y_best)
        x_start = x_grid[np.argmax(ei_vals)]

        res = _scipy_minimize(
            fun=lambda x: -float(_ei(np.array([x[0]]), gp_surr, y_best)[0]),
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
    bo_time = time.time() - t0

    # -- Final model: train on full dataset with best lamb_last --
    t1 = time.time()
    model = ProfileWKN(
        width=width, grid=GRID, sigma=SIGMA,
        seed=seed, device=device, **model_kwargs,
    )
    results, _, _ = model.train(
        dataset, opt='Adam', steps=STEPS, lr=LR,
        lamb_last=best_lamb_last, lamb_lower=fixed_lamb_lower,
        batch=BATCH, update_grid=False,
        verbose=False, device=device,
        early_stopping=True, patience=50, min_delta=1e-5,
    )
    train_time = time.time() - t1

    # Refit last layer with BO-selected lamb_last
    model.fit_last_layer(
        dataset['train_input'], dataset['train_label'],
        lamb_last=best_lamb_last, device=device,
    )

    # GP posterior inference (used for both point prediction and BCI)
    y_true = dataset['test_true'].view(-1).to(device)
    y_obs_test = dataset['test_label'].view(-1).to(device)
    pred_mean, _, pred_std, info = model.inference(
        dataset, lamb_last=best_lamb_last, device=device,
    )

    # Point prediction RMSE
    rmse = float(torch.sqrt(torch.mean((y_true - pred_mean) ** 2)))

    # Function CI: covers f(x*) (no noise)
    cover_f = (torch.abs(pred_mean - y_true) < 1.96 * pred_std)
    bci_coverage = float(torch.mean(cover_f.float()))
    mean_ci_width = float(torch.mean(2 * 1.96 * pred_std))

    # Predictive CI: covers y* = f(x*) + e (adds sigma^2 to variance)
    sigma2 = info['noise_var']
    pred_std_y = torch.sqrt(pred_std ** 2 + sigma2)
    cover_y = (torch.abs(pred_mean - y_obs_test) < 1.96 * pred_std_y)
    pred_coverage = float(torch.mean(cover_y.float()))
    pred_ci_width = float(torch.mean(2 * 1.96 * pred_std_y))

    # Width at matched coverage (95%)
    y_resid = torch.abs(pred_mean - y_obs_test).detach().cpu().numpy()
    y_std_np = pred_std_y.detach().cpu().numpy()
    y_width_at_95 = compute_width_at_coverage(y_resid, y_std_np, 0.95)

    return {
        'method': method_name,
        'lamb_last': best_lamb_last,
        'lamb_lower': fixed_lamb_lower,
        'rmse': rmse,
        'bci_coverage': bci_coverage,
        'mean_ci_width': mean_ci_width,
        'pred_coverage': pred_coverage,
        'pred_ci_width': pred_ci_width,
        'y_width_at_95': y_width_at_95,
        'noise_var': info['noise_var'],
        'pred_mean': pred_mean.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
        'train_loss': results['train_loss'],
        'test_loss': results['test_loss'],
        'test_loss2': results.get('test_loss2', results['test_loss']),
        'profile_obj': results.get('profile_obj', []),
        'bo_time': bo_time,
        'train_time': train_time,
        'model': model,
    }


def run_wahkon(exp_cfg, dataset, n_train, seed, device):
    """Wahkon (profile) with additive kernel, fixed lamb_lower = scale."""
    return _run_wahkon_core(
        exp_cfg, dataset, n_train, seed, device,
        method_name='Wahkon',
    )


# Backward-compatible aliases
run_profile_wkn = run_wahkon
_run_profile_wkn_core = _run_wahkon_core


# ============================================================
#  MLP baseline  (RMSE only)
# ============================================================

class _MLP(nn.Module):
    """Simple feedforward MLP with ReLU activations."""
    def __init__(self, in_dim, hidden_dims, out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _mlp_hidden_dims(width, grid_size=9):
    """Derive MLP hidden-layer sizes from the Wahkon width list.

    Maintains the same depth (number of hidden layers) and scales each
    hidden layer's neuron count by sqrt(grid_size) to roughly match the
    parameter count of a Wahkon model with the given grid size.

    E.g. width=[2,5,1], grid_size=9 -> hidden_dims=[15]  (5x3 = 15)
         width=[4,4,4,1], grid_size=9 -> hidden_dims=[12, 12]
    """
    import math
    scale = math.sqrt(grid_size)
    return [max(1, int(round(w * scale))) for w in width[1:-1]]


def _mlp_deep_hidden_dims(width):
    """Derive a higher-capacity MLP architecture from the WKN width list.

    10x width and 2x depth: each hidden layer's neuron count is scaled
    by 10, and each hidden layer is duplicated to double the depth.

    E.g. width=[2,5,1]       -> hidden_dims=[50, 50]
         width=[2,5,4,1]     -> hidden_dims=[50, 50, 40, 40]
         width=[4,4,4,1]     -> hidden_dims=[40, 40, 40, 40]
         width=[4,4,2,1]     -> hidden_dims=[40, 40, 20, 20]
         width=[10,10,10,1]  -> hidden_dims=[100, 100, 100, 100, 100, 100]
    """
    hidden = []
    for w in width[1:-1]:
        scaled = w * 10
        hidden.extend([scaled, scaled])
    return hidden


def run_mlp(exp_cfg, dataset, n_train, seed, device):
    """Train a standard MLP with early stopping and report RMSE (no CI)."""
    torch.manual_seed(seed)
    n_var = exp_cfg['n_var']
    hidden = _mlp_hidden_dims(exp_cfg['width'])

    model = _MLP(n_var, hidden, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1, 1)
    x_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device).view(-1, 1)

    t0 = time.time()
    batch_sz = min(BATCH, n_train)
    patience = 50
    best_test_loss = float('inf')
    best_state = None
    steps_no_improve = 0

    for step in range(STEPS):
        idx = np.random.choice(n_train, batch_sz, replace=False)
        xb, yb = x_train[idx], y_train[idx]
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check test loss every 10 steps
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                test_pred = model(x_test)
                test_loss = float(torch.mean((test_pred - y_test) ** 2))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_no_improve = 0
            else:
                steps_no_improve += 10
            if steps_no_improve >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    # Evaluate on test set
    with torch.no_grad():
        y_pred = model(x_test).view(-1)
        y_true = dataset['test_true'].view(-1).to(device)
        rmse = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))

    return {
        'method': 'MLP',
        'rmse': rmse,
        'bci_coverage': float('nan'),
        'mean_ci_width': float('nan'),
        'y_width_at_95': float('nan'),
        'noise_var': float('nan'),
        'pred_mean': y_pred.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
        'bo_time': 0.0,
        'train_time': train_time,
        'model': model,
    }


def run_mlp_deep(exp_cfg, dataset, n_train, seed, device):
    """Train a deeper, wider MLP (2x width, 2x depth) and report RMSE."""
    torch.manual_seed(seed)
    n_var = exp_cfg['n_var']
    hidden = _mlp_deep_hidden_dims(exp_cfg['width'])

    model = _MLP(n_var, hidden, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1, 1)
    x_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device).view(-1, 1)

    t0 = time.time()
    batch_sz = min(BATCH, n_train)
    patience = 50
    best_test_loss = float('inf')
    best_state = None
    steps_no_improve = 0

    for step in range(STEPS):
        idx = np.random.choice(n_train, batch_sz, replace=False)
        xb, yb = x_train[idx], y_train[idx]
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check test loss every 10 steps
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                test_pred = model(x_test)
                test_loss = float(torch.mean((test_pred - y_test) ** 2))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_no_improve = 0
            else:
                steps_no_improve += 10
            if steps_no_improve >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    # Evaluate on test set
    with torch.no_grad():
        y_pred = model(x_test).view(-1)
        y_true = dataset['test_true'].view(-1).to(device)
        rmse = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))

    return {
        'method': 'MLP (Deep)',
        'rmse': rmse,
        'bci_coverage': float('nan'),
        'mean_ci_width': float('nan'),
        'y_width_at_95': float('nan'),
        'noise_var': float('nan'),
        'pred_mean': y_pred.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
        'bo_time': 0.0,
        'train_time': train_time,
        'model': model,
    }


# ============================================================
#  Bayesian NN baseline (Bayes by Backprop)  -- RMSE + CI
#  Reference: Blundell et al. (2015), "Weight Uncertainty in
#  Neural Networks", ICML.
# ============================================================

class _BayesLinear(nn.Module):
    """Linear layer with mean-field Gaussian variational posterior.

    Each weight w_ij has parameters (mu, rho) where
        sigma = log(1 + exp(rho))   (softplus)
        w = mu + sigma * epsilon,   epsilon ~ N(0,1)

    The KL divergence KL(q || p) is computed analytically against a
    zero-mean Gaussian prior N(0, prior_sigma^2).
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(
            torch.empty(out_features).uniform_(-5, -4))

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        """Analytic KL(q || p) for Gaussian q and Gaussian prior."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        prior_var = self.prior_sigma ** 2

        kl_w = 0.5 * torch.sum(
            (weight_sigma ** 2 + self.weight_mu ** 2) / prior_var
            - 1.0
            - 2.0 * torch.log(weight_sigma / self.prior_sigma)
        )
        kl_b = 0.5 * torch.sum(
            (bias_sigma ** 2 + self.bias_mu ** 2) / prior_var
            - 1.0
            - 2.0 * torch.log(bias_sigma / self.prior_sigma)
        )
        return kl_w + kl_b


class _BNN_BBB(nn.Module):
    """Bayesian NN with Bayes by Backprop (mean-field Gaussian)."""
    def __init__(self, in_dim, hidden_dims, out_dim=1, prior_sigma=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_dim
        for h in hidden_dims:
            self.layers.append(_BayesLinear(prev, h, prior_sigma))
            prev = h
        self.layers.append(_BayesLinear(prev, out_dim, prior_sigma))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.layers)


def run_bnn(exp_cfg, dataset, n_train, seed, device,
            n_mc=50, prior_sigma=1.0):
    """
    Bayesian NN via Bayes by Backprop (Blundell et al., 2015).

    Same architecture as MLP (same hidden dims from _mlp_hidden_dims).
    Trained by minimising the ELBO:
        L = (n/B) * MSE(batch) + (1/n) * KL(q || prior)
    where B is the batch size and the KL is scaled by 1/n following
    the standard minibatch ELBO.

    At test time, n_mc forward passes with sampled weights provide
    the predictive mean and uncertainty.

    Provides RMSE + CI (coverage and width).
    """
    torch.manual_seed(seed)
    n_var = exp_cfg['n_var']
    hidden = _mlp_hidden_dims(exp_cfg['width'])

    model = _BNN_BBB(n_var, hidden, 1, prior_sigma=prior_sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_train = dataset['train_input'].to(device)
    y_train = dataset['train_label'].to(device).view(-1, 1)
    x_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device).view(-1, 1)

    # Train with ELBO + early stopping
    t0 = time.time()
    batch_sz = min(BATCH, n_train)
    patience = 50
    best_test_loss = float('inf')
    best_state = None
    steps_no_improve = 0

    for step in range(STEPS):
        model.train()
        idx = np.random.choice(n_train, batch_sz, replace=False)
        xb, yb = x_train[idx], y_train[idx]
        pred = model(xb)

        # ELBO: data fit (scaled to full dataset) + KL (scaled by 1/n)
        nll = (n_train / batch_sz) * torch.mean((pred - yb) ** 2)
        kl = model.kl_divergence() / n_train
        loss = nll + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            # Evaluate with mean weights (no sampling) for early stopping
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test)
                test_loss = float(torch.mean((test_pred - y_test) ** 2))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                steps_no_improve = 0
            else:
                steps_no_improve += 10
            if steps_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    # Posterior predictive: sample weights n_mc times
    model.train()  # enable weight sampling
    mc_preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            mc_preds.append(model(x_test).view(-1))
    mc_preds = torch.stack(mc_preds, dim=0)  # (n_mc, n_test)

    pred_mean = mc_preds.mean(dim=0)
    pred_std = mc_preds.std(dim=0)

    y_true = dataset['test_true'].view(-1).to(device)
    y_obs_test = dataset['test_label'].view(-1).to(device)

    rmse = float(torch.sqrt(torch.mean((pred_mean - y_true) ** 2)))

    # Function CI: covers f(x*)
    cover_f = (torch.abs(pred_mean - y_true) < 1.96 * pred_std)
    bci_coverage = float(torch.mean(cover_f.float()))
    mean_ci_width = float(torch.mean(2 * 1.96 * pred_std))

    # Estimate noise variance from training residuals (using mean weights)
    model.eval()
    with torch.no_grad():
        f_train = model(x_train).view(-1)
    resid = y_train.view(-1) - f_train
    sigma2 = float(torch.sum(resid ** 2) / max(n_train - 1, 1))

    # Predictive CI: Var(y*) = Var_MC(f*) + sigma^2
    pred_std_y = torch.sqrt(pred_std ** 2 + sigma2)
    cover_y = (torch.abs(pred_mean - y_obs_test) < 1.96 * pred_std_y)
    pred_coverage = float(torch.mean(cover_y.float()))
    pred_ci_width = float(torch.mean(2 * 1.96 * pred_std_y))

    # Width at matched coverage
    y_resid = torch.abs(pred_mean - y_obs_test).detach().cpu().numpy()
    y_std_np = pred_std_y.detach().cpu().numpy()
    y_width_at_95 = compute_width_at_coverage(y_resid, y_std_np, 0.95)

    return {
        'method': 'BNN',
        'rmse': rmse,
        'bci_coverage': bci_coverage,
        'mean_ci_width': mean_ci_width,
        'pred_coverage': pred_coverage,
        'pred_ci_width': pred_ci_width,
        'y_width_at_95': y_width_at_95,
        'noise_var': sigma2,
        'pred_mean': pred_mean.detach().cpu().numpy(),
        'y_true': y_true.detach().cpu().numpy(),
        'bo_time': 0.0,
        'train_time': train_time,
        'model': model,
    }


# ============================================================
#  NTK MLP class  (needed by run_calibrated_ci.py)
# ============================================================

class _NTK_MLP(nn.Module):
    """Wide MLP with Erf activation for NTK computation."""
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        # Initialise: W_std=1.5, b_std=0.05 (matching NTK_exp.py)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, std=1.5 / (layer.weight.shape[1] ** 0.5))
            nn.init.normal_(layer.bias, std=0.05)

    def forward(self, x):
        x = torch.erf(self.fc1(x))
        x = torch.erf(self.fc2(x))
        return self.fc3(x)
