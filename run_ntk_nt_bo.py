#!/usr/bin/env python3
"""
run_ntk_nt_bo.py — NTK experiments with BO-tuned noise variance.

Same as run_ntk_nt.py but selects σ² via Bayesian optimisation (GP
surrogate + Expected Improvement acquisition) minimising the negative
GP marginal log-likelihood, then calls
``nt.predict.gradient_descent_mse_ensemble`` for NNGP posterior inference.

Method display name in CSV: "NTK"

Supports parallel execution across seeds via --n_jobs (matching run_repeated.py).

Setup
-----
    python3.10 -m venv ntk_env
    source ntk_env/bin/activate
    pip install jax jaxlib neural-tangents numpy pandas filelock torch scipy scikit-learn

Usage
-----
    source ntk_env/bin/activate

    # Run all default experiments with 100 seeds (parallel on all cores):
    python run_ntk_nt_bo.py --out_dir results/run1

    # Run specific experiments:
    python run_ntk_nt_bo.py --expname f1 f3 --n_train 100 200 400 --out_dir results/run1 --n_jobs 4

    # Override existing NTK results:
    python run_ntk_nt_bo.py --override --n_seeds 20 --n_jobs 20 --out_dir results/simu --expname f3 f1

    # Custom BO settings:
    python run_ntk_nt_bo.py --expname f1 --n_train 100 --bo_calls 20 --sigma2_lo 1e-8 --sigma2_hi 100 --out_dir results/run1
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import filelock
from multiprocessing import Pool, cpu_count
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm as _scipy_norm
from scipy.optimize import minimize as _scipy_minimize

# ============================================================
#  Dataset creation (self-contained, matches compare_methods.py)
# ============================================================
TEST_NUM = 1000

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


# ============================================================
#  Utility: Width at matched coverage
# ============================================================

def compute_width_at_coverage(residuals, pred_std, target_coverage=0.95):
    """Compute the average CI width needed to achieve a target empirical coverage."""
    residuals = np.asarray(residuals, dtype=np.float64)
    pred_std = np.asarray(pred_std, dtype=np.float64)

    if len(residuals) == 0 or np.any(pred_std <= 0):
        return float('nan')

    z_scores = residuals / pred_std
    z_target = float(np.quantile(z_scores, target_coverage))
    width = float(np.mean(2 * z_target * pred_std))
    return width


# ============================================================
#  Experiment configurations  (from inference.py)
# ============================================================
EXPERIMENTS = {
    # f1 (was exp5) — log + cot, D=3
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
    # f2 (was sin_sum_sq) — sin of sum of squares, D=10
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
        'description': 'sin(x₁²+x₂²+...+x₁₀²): smooth links, rapid oscillation in 10D',
    },
    # f3 (was exp3) — exp of paired sin, D=4
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
        'noise_std': 0.1#0.420433/2,
    },
    # f4 (was nested_comp) — deep nested composition, D=6
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
        'description': 'exp(sin(π(x₁²+x₂²)))·cos(πx₃x₄): deep composition + noise vars',
    },
}

# Backward-compat aliases
EXPERIMENTS['exp5'] = EXPERIMENTS['f1']
EXPERIMENTS['sin_sum_sq'] = EXPERIMENTS['f2']
EXPERIMENTS['exp3'] = EXPERIMENTS['f3']
EXPERIMENTS['nested_comp'] = EXPERIMENTS['f4']



# ============================================================
#  NTK runner — analytic infinite-width NTK
# ============================================================

DISPLAY_NAME = 'NTK'

# Module-level config (set from CLI args in main, read by workers)
_NTK_CONFIG = {
    'activation': 'erf',
    'sigma2_range': (1e-4, 10.0),
    'bo_calls': 15,
    'bo_random_init': 5,
    'bo_xi': 0.01,
}


def _gp_neg_log_marginal_likelihood(K_tt, y_train, sigma2):
    """Compute negative log marginal likelihood for GP with NTK kernel.

    log p(y|σ²) = -½ yᵀ (K + σ²I)⁻¹ y - ½ log|K + σ²I| - n/2 log(2π)
    """
    n = len(y_train)
    reg_K = K_tt + sigma2 * np.eye(n)
    try:
        L = np.linalg.cholesky(reg_K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        data_fit = 0.5 * float(y_train @ alpha)
        complexity = float(np.sum(np.log(np.diag(L))))
        nlml = data_fit + complexity + 0.5 * n * np.log(2 * np.pi)
    except np.linalg.LinAlgError:
        nlml = 1e12
    return float(nlml)


def run_ntk_nt(exp_cfg, dataset, n_train, seed):
    """
    Analytic (infinite-width) NTK using the neural-tangents library.

    Builds a fully-connected network matching the Wahkon architecture,
    selects the noise variance σ² via Bayesian optimisation (GP surrogate
    + Expected Improvement) minimising the neg log marginal likelihood,
    and uses ``nt.predict.gradient_descent_mse_ensemble`` for NNGP
    posterior predictions and confidence intervals.
    """
    import jax
    import jax.numpy as jnp
    import neural_tangents as nt
    from neural_tangents import stax

    cfg = _NTK_CONFIG
    np.random.seed(seed)

    # ── Data ────────────────────────────────────────────────────
    x_train = dataset['train_input'].cpu().numpy().astype(np.float64)
    y_train_1d = dataset['train_label'].cpu().numpy().ravel().astype(np.float64)
    y_train_2d = y_train_1d[:, None]  # [n, 1] for neural-tangents
    x_test  = dataset['test_input'].cpu().numpy().astype(np.float64)
    y_true  = dataset['test_true'].cpu().numpy().ravel().astype(np.float64)
    y_obs_test = dataset['test_label'].cpu().numpy().ravel().astype(np.float64)

    # ── Build architecture matching Wahkon width ────────────────
    width = exp_cfg['width']
    activation = cfg['activation']
    act_fn = stax.Erf() if activation == 'erf' else stax.Relu()

    layers = []
    for i, w in enumerate(width[1:]):
        layers.append(stax.Dense(w, W_std=1.5, b_std=0.05))
        if i < len(width) - 2:
            layers.append(act_fn)

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    # ── Compute NTK train-train kernel ─────────────────────────
    t0 = time.time()

    x_tr_jax = jnp.array(x_train)
    x_te_jax = jnp.array(x_test)

    K_tt = np.array(kernel_fn(x_tr_jax, x_tr_jax, 'ntk'))

    # ── BO for noise variance σ² ──────────────────────────────
    lo, hi = cfg['sigma2_range']
    _PENALTY = 1e12
    rng = np.random.RandomState(seed)

    def _objective(sigma2_val):
        return _gp_neg_log_marginal_likelihood(K_tt, y_train_1d, sigma2_val)

    # EI acquisition function
    def _ei(X_cand, gp_surr, y_best):
        mu, std = gp_surr.predict(X_cand.reshape(-1, 1), return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - cfg['bo_xi']) / std
        return ((y_best - mu - cfg['bo_xi']) * _scipy_norm.cdf(Z)
                + std * _scipy_norm.pdf(Z))

    # Phase 1: random initialisation (log-uniform)
    n_random = min(cfg['bo_random_init'], cfg['bo_calls'])
    X_obs = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_random, 1)))
    y_obs = np.array([_objective(x[0]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # GP surrogate
    length_scale = 0.4 * (hi - lo + 1e-6)
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=(0.05 * (hi - lo + 1e-6),
                             10.0 * (hi - lo + 1e-6)),
        nu=2.5,
    )
    gp_surr = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-3, normalize_y=True,
        n_restarts_optimizer=5, random_state=seed,
    )

    # Phase 2: BO loop
    for _ in range(cfg['bo_calls'] - n_random):
        gp_surr.fit(X_obs, y_obs)
        y_best = float(y_obs.min())

        x_grid = np.linspace(lo, hi, 200).reshape(-1, 1)
        ei_vals = _ei(x_grid.ravel(), gp_surr, y_best)
        x_start = x_grid[np.argmax(ei_vals)]

        res = _scipy_minimize(
            fun=lambda x: -float(_ei(np.array([x[0]]), gp_surr, y_best)[0]),
            x0=x_start, bounds=[(lo, hi)], method='L-BFGS-B',
        )
        x_next = np.clip(
            res.x if res.success else x_start.ravel(), lo, hi,
        ).reshape(1, 1)

        y_next = _objective(x_next[0, 0])
        y_next = y_next if np.isfinite(y_next) else _PENALTY
        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

    best_idx = int(np.argmin(y_obs))
    sigma2 = float(X_obs[best_idx, 0])

    # ── Predictions via gradient_descent_mse_ensemble ──────────
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn, x_tr_jax, y_train_2d, diag_reg=sigma2,
    )

    nngp_mean, nngp_covariance = predict_fn(
        x_test=x_te_jax, get='nngp', compute_cov=True,
    )
    pred_mean = jnp.reshape(nngp_mean, (-1,))
    pred_std = jnp.sqrt(jnp.diag(nngp_covariance))

    pred_mean = np.array(pred_mean)
    pred_std = np.array(pred_std)
    pred_var = pred_std ** 2
    train_time = time.time() - t0

    # ── Metrics ─────────────────────────────────────────────────
    rmse = float(np.sqrt(np.mean((pred_mean - y_true) ** 2)))

    f_resid = np.abs(pred_mean - y_true)
    cover = f_resid < 1.96 * pred_std
    bci_coverage = float(np.mean(cover))
    mean_ci_width = float(np.mean(2 * 1.96 * pred_std))

    pred_std_y = np.sqrt(pred_var + sigma2)
    pred_cover = np.abs(pred_mean - y_obs_test) < 1.96 * pred_std_y
    pred_coverage = float(np.mean(pred_cover))
    pred_ci_width = float(np.mean(2 * 1.96 * pred_std_y))

    y_resid = np.abs(pred_mean - y_obs_test)
    y_width_at_95 = compute_width_at_coverage(y_resid, pred_std_y, 0.95)

    return {
        'method': DISPLAY_NAME,
        'rmse': rmse,
        'bci_coverage': bci_coverage,
        'mean_ci_width': mean_ci_width,
        'pred_coverage': pred_coverage,
        'pred_ci_width': pred_ci_width,
        'y_width_at_95': y_width_at_95,
        'sigma2': sigma2,
        'train_time': train_time,
    }


# ============================================================
#  Worker / log helpers (matching run_repeated.py)
# ============================================================

def _worker_fn(args):
    """Worker function for parallel execution. Runs one seed, returns rows."""
    expname, n_train, seed, device = args
    exp_cfg = EXPERIMENTS[expname]
    exp_ranges = exp_cfg.get('ranges', [-1, 1])

    t0 = time.time()
    try:
        dataset = create_dataset(
            exp_cfg['f'], f_true=exp_cfg['f_true'],
            n_var=exp_cfg['n_var'], ranges=exp_ranges,
            train_num=n_train, test_num=TEST_NUM,
            normalize_input=True, normalize_label=True,
            seed=seed, device=device,
        )
        res = run_ntk_nt(exp_cfg, dataset, n_train, seed)
        seed_rows = [{
            'seed': seed,
            'method': res['method'],
            'rmse': res['rmse'],
            'y_width_at_95': res.get('y_width_at_95', np.nan),
        }]
    except Exception as e:
        print(f"  Seed {seed} FAILED: {e}")
        seed_rows = []

    elapsed = time.time() - t0
    return seed, seed_rows, elapsed


def _append_to_log(log_path, seed_rows):
    """Thread-safe append of seed results to the CSV log."""
    df_seed = pd.DataFrame(seed_rows)
    lock_path = log_path + '.lock'
    lock = filelock.FileLock(lock_path, timeout=60)
    with lock:
        write_header = not os.path.exists(log_path)
        df_seed.to_csv(log_path, mode='a', header=write_header, index=False)


def _print_summary(log_path, expname, n_train):
    """Print aggregate summary table from log CSV (matching run_repeated.py)."""
    if not os.path.exists(log_path):
        print("  No results to aggregate.")
        return

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {expname}, n_train={n_train}")
    print(f"{'=' * 70}")

    df_all = pd.read_csv(log_path)
    n_seeds_done = df_all['seed'].nunique()
    all_methods = sorted(df_all['method'].unique())

    rows = []
    for mname in all_methods:
        mdf = df_all[df_all['method'] == mname]
        rows.append({
            'Method': mname,
            'RMSE (mean)': mdf['rmse'].mean(),
            'RMSE (std)': mdf['rmse'].std(),
            'y-W@95 (mean)': mdf['y_width_at_95'].mean() if 'y_width_at_95' in mdf.columns else np.nan,
            'y-W@95 (std)': mdf['y_width_at_95'].std() if 'y_width_at_95' in mdf.columns else np.nan,
            'N runs': len(mdf),
        })

    df_summary = pd.DataFrame(rows)

    print(f"\n{'Method':<26} {'RMSE':>16} {'y-W@95':>16} {'N':>3}")
    print('-' * 65)
    for _, row in df_summary.iterrows():
        rmse_str = f"{row['RMSE (mean)']:.4f}±{row['RMSE (std)']:.4f}"
        yw95_str = f"{row['y-W@95 (mean)']:.4f}±{row['y-W@95 (std)']:.4f}" if not np.isnan(row.get('y-W@95 (mean)', np.nan)) else "N/A"
        print(f"{row['Method']:<26} {rmse_str:>16} {yw95_str:>16} {row['N runs']:>3.0f}")

    out_dir = os.path.dirname(log_path)
    summary_path = os.path.join(out_dir, f'{expname}_n{n_train}_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Summary: {summary_path}")
    print(f"  Log:     {log_path} ({n_seeds_done} seeds)")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run repeated NTK (neural-tangents) experiments (parallel)',
    )
    parser.add_argument('--expname', type=str, nargs='+',
                        default=['f1', 'f2', 'f3', 'f4'],
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiments to run (default: f1 f2 f3 f4)')
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='Number of random seeds (default: 100)')
    parser.add_argument('--seed_start', type=int, default=1,
                        help='Starting seed (default: 1)')
    parser.add_argument('--n_train', type=int, nargs='+',
                        default=[100, 200, 400, 800, 1600, 3200, 10000],
                        help='Training sizes (default: 100 200 400 800 1600 3200 10000)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for dataset creation (NTK uses JAX/CPU)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for results and checkpoint logs')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (default: -1 = all CPU cores). '
                             'Use 1 for sequential execution.')
    parser.add_argument('--override', action='store_true',
                        help='Remove existing NTK rows and re-run')
    parser.add_argument('--activation', type=str, default='erf',
                        choices=['erf', 'relu'],
                        help='Activation function (default: erf)')
    parser.add_argument('--sigma2_lo', type=float, default=1e-4,
                        help='Lower bound for sigma2 BO search (default: 1e-4)')
    parser.add_argument('--sigma2_hi', type=float, default=10.0,
                        help='Upper bound for sigma2 BO search (default: 10.0)')
    parser.add_argument('--bo_calls', type=int, default=15,
                        help='Total BO iterations (default: 15)')

    args = parser.parse_args()

    # Propagate config to module-level (for workers)
    _NTK_CONFIG['activation'] = args.activation
    _NTK_CONFIG['sigma2_range'] = (args.sigma2_lo, args.sigma2_hi)
    _NTK_CONFIG['bo_calls'] = args.bo_calls

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Resolve n_jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = max(1, cpu_count() + n_jobs)
    n_jobs = min(n_jobs, len(seeds))

    print(f"{'=' * 70}")
    print(f"  NTK (neural-tangents) Repeated Experiment Runner (parallel)")
    print(f"  Experiments: {args.expname}")
    print(f"  n_train: {args.n_train}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({args.n_seeds} total)")
    print(f"  Activation: {args.activation}")
    print(f"  sigma2 BO range: [{args.sigma2_lo}, {args.sigma2_hi}], bo_calls={args.bo_calls}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")

    for expname in args.expname:
        exp_cfg = EXPERIMENTS[expname]
        desc = exp_cfg.get('description', '')

        for n_train in args.n_train:
            print(f"\n{'#' * 70}")
            print(f"  {expname}: n_var={exp_cfg['n_var']}, n_train={n_train}, "
                  f"noise_std={exp_cfg['noise_std']}")
            if desc:
                print(f"  {desc}")
            print(f"{'#' * 70}")

            log_path = os.path.join(out_dir, f'{expname}_n{n_train}_log.csv')

            # --override: remove existing NTK rows
            if args.override and os.path.exists(log_path):
                lock_path = log_path + '.lock'
                lock = filelock.FileLock(lock_path, timeout=60)
                with lock:
                    try:
                        df_existing = pd.read_csv(log_path)
                        n_before = len(df_existing)
                        df_existing = df_existing[df_existing['method'] != DISPLAY_NAME]
                        n_removed = n_before - len(df_existing)
                        if n_removed > 0:
                            if len(df_existing) > 0:
                                df_existing.to_csv(log_path, index=False)
                            else:
                                os.remove(log_path)
                            print(f"  Override: removed {n_removed} rows for {DISPLAY_NAME}")
                        else:
                            print(f"  Override: no existing rows to remove for {DISPLAY_NAME}")
                    except Exception:
                        pass

            # Check for already-completed seeds
            completed_seeds = set()
            if os.path.exists(log_path):
                try:
                    df_existing = pd.read_csv(log_path)
                    for s in df_existing['seed'].unique():
                        seed_methods = set(df_existing[df_existing['seed'] == s]['method'].values)
                        if DISPLAY_NAME in seed_methods:
                            completed_seeds.add(s)
                    if completed_seeds:
                        print(f"  Resuming: {len(completed_seeds)} seeds already have {DISPLAY_NAME}")
                except Exception:
                    completed_seeds = set()

            pending_seeds = [s for s in seeds if s not in completed_seeds]
            if not pending_seeds:
                print(f"  All {len(seeds)} seeds already completed. Skipping.")
            elif n_jobs == 1:
                # Sequential execution
                exp_ranges = exp_cfg.get('ranges', [-1, 1])
                for i, seed in enumerate(pending_seeds):
                    t0 = time.time()
                    idx = seeds.index(seed)
                    print(f"\n  Seed {seed} ({idx+1}/{args.n_seeds}) ...", end='', flush=True)

                    dataset = create_dataset(
                        exp_cfg['f'], f_true=exp_cfg['f_true'],
                        n_var=exp_cfg['n_var'], ranges=exp_ranges,
                        train_num=n_train, test_num=TEST_NUM,
                        normalize_input=True, normalize_label=True,
                        seed=seed, device=args.device,
                    )

                    try:
                        res = run_ntk_nt(exp_cfg, dataset, n_train, seed)
                        print(f" RMSE={res['rmse']:.4f}", end='')

                        seed_rows = [{
                            'seed': seed, 'method': res['method'],
                            'rmse': res['rmse'],
                            'y_width_at_95': res.get('y_width_at_95', np.nan),
                        }]
                        _append_to_log(log_path, seed_rows)
                    except Exception as e:
                        print(f" FAILED: {e}", end='')

                    elapsed = time.time() - t0
                    print(f" ({elapsed:.1f}s)")
            else:
                # Parallel execution
                print(f"  Running {len(pending_seeds)} seeds with {n_jobs} workers...")
                worker_args = [
                    (expname, n_train, seed, args.device)
                    for seed in pending_seeds
                ]

                t_start = time.time()
                with Pool(processes=n_jobs) as pool:
                    results = pool.map(_worker_fn, worker_args)

                for seed, seed_rows, elapsed in sorted(results, key=lambda x: x[0]):
                    if seed_rows:
                        _append_to_log(log_path, seed_rows)
                    rmse_str = f"RMSE={seed_rows[0]['rmse']:.4f}" if seed_rows else "FAILED"
                    print(f"    Seed {seed}: {rmse_str} ({elapsed:.1f}s)")

                total_time = time.time() - t_start
                print(f"  Parallel block done in {total_time:.1f}s "
                      f"(avg {total_time/len(pending_seeds):.1f}s/seed)")

            # Print summary table
            _print_summary(log_path, expname, n_train)

    print(f"\nDone. All results in {out_dir}/")


if __name__ == '__main__':
    main()
