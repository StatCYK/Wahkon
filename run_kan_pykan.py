#!/usr/bin/env python3
"""
run_kan_pykan.py — Repeated-seed KAN experiments using the official pykan package.

This is a standalone script that runs the official pykan KAN (B-spline)
on the same experiments and datasets as compare_methods.py, writing results
to the same CSV log files so that plot_results.py can compare all methods.

Method display name in CSV: "KAN"

Supports parallel execution across seeds via --n_jobs (matching run_repeated.py).

Setup
-----
    # Create a separate Python 3.10 environment:
    python3.10 -m venv kan_env
    source kan_env/bin/activate
    pip install torch numpy pandas filelock pykan

Usage
-----
    source kan_env/bin/activate

    # Run all default experiments with 100 seeds (parallel on all cores):
    python run_kan_pykan.py --out_dir results/run1

    # Run specific experiments:
    python run_kan_pykan.py --expname f1 f2 --n_train 100 200 400 --out_dir results/run1 --n_jobs 4

    # Override existing KAN results:
    python run_kan_pykan.py --expname f4 --override --n_seeds 20 --out_dir results/run1

    # Customize KAN training:
    python run_kan_pykan.py --expname f1 --n_train 100 --grid 5 --steps 200 --lr 0.005 --out_dir results/run1
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
from sklearn.model_selection import KFold
from scipy.stats import norm as _scipy_norm
from scipy.optimize import minimize as _scipy_minimize

# ── Official pykan ──────────────────────────────────────────────
from kan import KAN

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
#  Experiment configs  (identical to compare_methods.py)
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
        'width': [3, 6, 6, 1],
        'n_var': 3,
        'noise_std': 0.1,
    },
    'f2': {
        'f': lambda x: (torch.sin(torch.sum(x ** 2, dim=1, keepdim=True))
                         + 0.1 * torch.randn(x.shape[0], 1)),
        'f_true': lambda x: torch.sin(torch.sum(x ** 2, dim=1, keepdim=True)),
        'width': [10, 10, 10, 1], 'n_var': 10, 'noise_std': 0.1, 'ranges': [-1, 1],
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
        'noise_std': 0.1,
    },
    'f4': {
        'f': lambda x: (torch.exp(torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2)))
                         * torch.cos(torch.pi * x[:, [2]] * x[:, [3]])
                         + 0.1 * torch.randn(x.shape[0], 1)),
        'f_true': lambda x: (torch.exp(torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2)))
                               * torch.cos(torch.pi * x[:, [2]] * x[:, [3]])),
        'width': [6, 6, 6, 6, 1], 'n_var': 6, 'noise_std': 0.1, 'ranges': [-1, 1],
    },
}

# Backward-compat aliases for old experiment names
EXPERIMENTS['exp5'] = EXPERIMENTS['f1']
EXPERIMENTS['sin_sum_sq'] = EXPERIMENTS['f2']
EXPERIMENTS['exp3'] = EXPERIMENTS['f3']
EXPERIMENTS['nested_comp'] = EXPERIMENTS['f4']


# ============================================================
#  KAN runner using official pykan
# ============================================================

DISPLAY_NAME = 'KAN'

# Module-level config (set from CLI args in main, read by workers)
_KAN_CONFIG = {
    'grid': 9, 'k': 3, 'steps': 200, 'lr': 0.005,
    'grid_update_freq': 10, 'stop_grid_update_step': 50,
    # BO settings for lamb selection
    'bo_calls': 15, 'bo_random_init': 5, 'bo_cv_folds': 5,
    'lamb_range': (1e-4, 1.0), 'bo_xi': 0.01,
    # Early stopping
    'patience': 50, 'min_delta': 1e-5,
}


def _train_kan_with_early_stopping(width, dataset, seed, device, cfg, lamb):
    """Train a KAN with early stopping on validation loss.

    Splits train data into 80/20, trains on the 80%, monitors val loss,
    and stops when val loss doesn't improve for `patience` steps.
    Returns the trained model.
    """
    n_total = dataset['train_input'].shape[0]
    perm = np.random.RandomState(seed).permutation(n_total)
    n_val = max(1, int(n_total * 0.2))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Create train-only dataset for KAN.fit()
    train_ds = {
        'train_input': dataset['train_input'][train_idx],
        'train_label': dataset['train_label'][train_idx],
        'test_input':  dataset['train_input'][val_idx],
        'test_label':  dataset['train_label'][val_idx],
    }

    model = KAN(width=width, grid=cfg['grid'], k=cfg['k'],
                seed=seed, device=device)

    # Train in chunks to implement early stopping
    patience = cfg['patience']
    min_delta = cfg['min_delta']
    chunk_size = 10  # evaluate val loss every chunk_size steps
    total_steps = cfg['steps']
    best_val_loss = float('inf')
    steps_no_improve = 0
    steps_done = 0

    while steps_done < total_steps:
        remaining = total_steps - steps_done
        chunk = min(chunk_size, remaining)
        # Remaining grid-update budget for this chunk
        remaining_grid_steps = max(0, cfg['stop_grid_update_step'] - steps_done)
        # Disable grid updates when budget is exhausted (avoids pykan
        # ZeroDivisionError: int(0 / grid_update_num) → freq=0 → step%0)
        do_grid_update = remaining_grid_steps > 0
        model.fit(
            train_ds, opt='Adam', lr=cfg['lr'], steps=chunk,
            lamb=lamb,
            update_grid=do_grid_update,
            grid_update_num=cfg['grid_update_freq'] if do_grid_update else 1,
            stop_grid_update_step=remaining_grid_steps if do_grid_update else 1,
        )
        steps_done += chunk

        # Evaluate validation loss
        with torch.no_grad():
            y_val_pred = model.forward(train_ds['test_input'].to(device)).view(-1)
            y_val_true = train_ds['test_label'].view(-1).to(device)
            val_loss = float(torch.mean((y_val_pred - y_val_true) ** 2))

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            steps_no_improve = 0
            # Save best state
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            steps_no_improve += chunk

        if steps_no_improve >= patience:
            break

    # Restore best state if we have one
    if 'best_state' in dir() and best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    return model


def run_kan_pykan(exp_cfg, dataset, n_train, seed, device):
    """
    Train the official pykan KAN with Bayesian optimization for lamb selection
    and early stopping, matching the WKN training protocol.

    BO selects lamb via K-fold CV validation RMSE. Final model is trained
    on full data with the best lamb and early stopping.
    """
    width = exp_cfg['width']
    cfg = _KAN_CONFIG

    torch.manual_seed(seed)
    np.random.seed(seed)

    lo, hi = cfg['lamb_range']
    _PENALTY = 1e6
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=cfg['bo_cv_folds'], shuffle=True, random_state=seed)

    # -- BO objective: K-fold CV validation RMSE for a given lamb --
    def _objective(lamb_val):
        val_losses = []
        for train_idx, val_idx in kf.split(range(n_train)):
            fold_ds = {
                'train_input': dataset['train_input'][train_idx],
                'train_label': dataset['train_label'][train_idx],
                'test_input':  dataset['train_input'][val_idx],
                'test_label':  dataset['train_label'][val_idx],
            }
            try:
                mdl = KAN(width=width, grid=cfg['grid'], k=cfg['k'],
                          seed=seed, device=device)
                _do_gu = cfg['stop_grid_update_step'] > 0
                mdl.fit(
                    fold_ds, opt='Adam', lr=cfg['lr'], steps=cfg['steps'],
                    lamb=lamb_val,
                    update_grid=_do_gu,
                    grid_update_num=cfg['grid_update_freq'] if _do_gu else 1,
                    stop_grid_update_step=cfg['stop_grid_update_step'] if _do_gu else 1,
                )
                with torch.no_grad():
                    y_val_pred = mdl.forward(fold_ds['test_input'].to(device)).view(-1)
                    y_val_true = fold_ds['test_label'].view(-1).to(device)
                    v = float(torch.sqrt(torch.mean((y_val_pred - y_val_true) ** 2)))
                val_losses.append(v if np.isfinite(v) else _PENALTY)
            except Exception:
                val_losses.append(_PENALTY)
        result = float(np.mean(val_losses))
        return result if np.isfinite(result) else _PENALTY

    # -- Expected Improvement acquisition function --
    def _ei(X_cand, gp_surr, y_best):
        mu, std = gp_surr.predict(X_cand.reshape(-1, 1), return_std=True)
        std = np.maximum(std, 1e-9)
        Z = (y_best - mu - cfg['bo_xi']) / std
        return (
            (y_best - mu - cfg['bo_xi']) * _scipy_norm.cdf(Z)
            + std * _scipy_norm.pdf(Z)
        )

    # -- Phase 1: random initialisation in log-space --
    t0 = time.time()
    n_random = min(cfg['bo_random_init'], cfg['bo_calls'])
    X_obs = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_random, 1)))
    y_obs = np.array([_objective(x[0]) for x in X_obs])
    y_obs = np.where(np.isfinite(y_obs), y_obs, _PENALTY)

    # -- GP surrogate (1-D Matérn-5/2) --
    length_scale = 0.4 * (hi - lo + 1e-6)
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=(0.05 * (hi - lo + 1e-6), 10.0 * (hi - lo + 1e-6)),
        nu=2.5,
    )
    gp_surr = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-3, normalize_y=True,
        n_restarts_optimizer=5, random_state=seed,
    )

    # -- Phase 2: BO loop --
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
    best_lamb = float(X_obs[best_idx, 0])
    bo_time = time.time() - t0

    # -- Final model: train on full dataset with best lamb + early stopping --
    t1 = time.time()
    model = _train_kan_with_early_stopping(
        width, dataset, seed, device, cfg, lamb=best_lamb,
    )
    train_time = time.time() - t1

    with torch.no_grad():
        y_pred = model.forward(dataset['test_input'].to(device)).view(-1)
        y_true = dataset['test_true'].view(-1).to(device)
        rmse = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))

    return {
        'method': DISPLAY_NAME,
        'rmse': rmse,
        'bci_coverage': float('nan'),
        'mean_ci_width': float('nan'),
        'pred_coverage': float('nan'),
        'pred_ci_width': float('nan'),
        'y_width_at_95': float('nan'),
        'noise_var': float('nan'),
        'lamb': best_lamb,
        'bo_time': bo_time,
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
        res = run_kan_pykan(exp_cfg, dataset, n_train, seed, device)
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
        description='Run repeated KAN (official pykan) experiments (parallel)',
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
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for results and checkpoint logs')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (default: -1 = all CPU cores). '
                             'Use 1 for sequential execution.')
    parser.add_argument('--override', action='store_true',
                        help='Remove existing KAN rows and re-run')

    # KAN hyperparameters
    parser.add_argument('--grid', type=int, default=9,
                        help='Grid intervals for B-spline (default: 9)')
    parser.add_argument('--k', type=int, default=3,
                        help='Spline order (default: 3)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Training steps (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate for Adam (default: 0.005)')
    parser.add_argument('--grid_update_freq', type=int, default=10)
    parser.add_argument('--stop_grid_update_step', type=int, default=50)
    # BO and early stopping
    parser.add_argument('--bo_calls', type=int, default=15,
                        help='Total BO evaluations for lamb selection (default: 15)')
    parser.add_argument('--bo_cv_folds', type=int, default=5,
                        help='K-fold CV folds for BO objective (default: 5)')
    parser.add_argument('--lamb_lo', type=float, default=1e-4,
                        help='Lower bound of lamb search range (default: 1e-4)')
    parser.add_argument('--lamb_hi', type=float, default=1.0,
                        help='Upper bound of lamb search range (default: 1.0)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience in steps (default: 50)')
    parser.add_argument('--min_delta', type=float, default=1e-5,
                        help='Early stopping min improvement (default: 1e-5)')

    args = parser.parse_args()

    # Propagate KAN hyperparameters to module-level config (for workers)
    _KAN_CONFIG.update({
        'grid': args.grid, 'k': args.k, 'steps': args.steps,
        'lr': args.lr,
        'grid_update_freq': args.grid_update_freq,
        'stop_grid_update_step': args.stop_grid_update_step,
        'bo_calls': args.bo_calls, 'bo_random_init': 5,
        'bo_cv_folds': args.bo_cv_folds,
        'lamb_range': (args.lamb_lo, args.lamb_hi),
        'bo_xi': 0.01,
        'patience': args.patience, 'min_delta': args.min_delta,
    })

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
    print(f"  KAN Repeated Experiment Runner (parallel)")
    print(f"  Experiments: {args.expname}")
    print(f"  n_train: {args.n_train}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({args.n_seeds} total)")
    print(f"  KAN config: grid={args.grid}, k={args.k}, steps={args.steps}, "
          f"lr={args.lr}")
    print(f"  BO: {args.bo_calls} calls, {args.bo_cv_folds}-fold CV, "
          f"lamb ∈ [{args.lamb_lo}, {args.lamb_hi}]")
    print(f"  Early stopping: patience={args.patience}, min_delta={args.min_delta}")
    print(f"  Device: {args.device}")
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

            # --override: remove existing KAN rows
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
                        res = run_kan_pykan(exp_cfg, dataset, n_train, seed, args.device)
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
