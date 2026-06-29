#!/usr/bin/env python
"""
run_repeated.py -- Run compare_methods experiments with multiple seeds.

Generates a fresh dataset for each seed, runs all enabled methods, and
reports mean +/- std of RMSE across repetitions.

Supports parallel execution across seeds via --n_jobs.

Usage
-----
    python run_repeated.py
    python run_repeated.py --expname f1 f3 --methods wahkon mlp
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import filelock  # pip install filelock

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from compare_methods import (
    EXPERIMENTS, GRID, SIGMA, STEPS, LR, BATCH, TEST_NUM,
    num_link_fun, create_dataset2,
    run_wahkon,
    run_mlp, run_mlp_deep, run_bnn,
)

# Method key -> (display name, runner function)
# KAN and NTK have separate scripts (run_kan_pykan.py, run_ntk_nt_bo.py)
METHOD_RUNNERS = {
    'wahkon':   ('Wahkon',     run_wahkon),
    'mlp':      ('MLP',        run_mlp),
    'mlp_deep': ('MLP (Deep)', run_mlp_deep),
    'bnn':      ('BNN',        run_bnn),
}

ALL_METHOD_KEYS = set(METHOD_RUNNERS.keys())
DEFAULT_METHODS = sorted(ALL_METHOD_KEYS)


def run_single_seed(expname, exp_cfg, n_train, seed, device, enabled_methods):
    """Run all enabled methods for one seed. Returns dict of results."""
    exp_ranges = exp_cfg.get('ranges', [-1, 1])

    dataset = create_dataset2(
        exp_cfg['f'], f_true=exp_cfg['f_true'],
        n_var=exp_cfg['n_var'], ranges=exp_ranges,
        train_num=n_train, test_num=TEST_NUM,
        normalize_input=True, normalize_label=True,
        seed=seed, device=device,
    )

    methods = {}

    for key in ['wahkon', 'mlp', 'mlp_deep', 'bnn']:
        if key not in enabled_methods:
            continue
        display_name, runner = METHOD_RUNNERS[key]
        try:
            res = runner(exp_cfg, dataset, n_train, seed, device)
            methods[display_name] = res
        except Exception as e:
            print(f"    {display_name} failed (seed={seed}): {e}")

    return methods


def _worker_fn(args):
    """Worker function for parallel execution. Runs one seed and returns rows."""
    expname, exp_cfg_key, n_train, seed, device, enabled_methods = args

    # Re-fetch exp_cfg from EXPERIMENTS (lambda functions can't be pickled directly,
    # but module-level EXPERIMENTS dict is available after import)
    exp_cfg = EXPERIMENTS[exp_cfg_key]

    t0 = time.time()
    try:
        methods = run_single_seed(expname, exp_cfg, n_train, seed, device, enabled_methods)
    except Exception as e:
        print(f"  Seed {seed} FAILED entirely: {e}")
        return seed, [], time.time() - t0

    seed_rows = []
    for mname, res in methods.items():
        seed_rows.append({
            'seed': seed,
            'method': mname,
            'rmse': res.get('rmse', np.nan),
            'y_width_at_95': res.get('y_width_at_95', np.nan),
        })

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


def main():
    parser = argparse.ArgumentParser(
        description='Run repeated experiments with multiple seeds (parallel)',
    )
    parser.add_argument('--expname', type=str, nargs='+',
                        default=['f1', 'f2', 'f3', 'f4'],
                        help='Experiments to run (default: f1 f2 f3 f4)')
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='Number of random seeds (default: 100)')
    parser.add_argument('--seed_start', type=int, default=1,
                        help='Starting seed (default: 1)')
    parser.add_argument('--n_train', type=int, nargs='+',
                        default=[100, 200, 400, 800, 1600, 3200, 10000],
                        help='Training sizes to run (default: 100 200 400 800 1600 3200 10000)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help=f'Methods to run (default: all). Choices: {sorted(ALL_METHOD_KEYS)}')
    parser.add_argument('--out_dir', type=str, default='./results/simu',
                        help='Output directory for results and checkpoint logs (default: ./results/simu)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (default: -1 = all CPUs). '
                             'Use 1 for sequential execution.')
    parser.add_argument('--override', action='store_true',
                        help='Override existing results for the specified --methods. '
                             'Removes old rows for those methods from the log CSV '
                             'and re-runs all seeds. Other methods are untouched.')
    args = parser.parse_args()

    # Enabled methods
    if args.methods is None:
        enabled_methods = ALL_METHOD_KEYS
    else:
        enabled_methods = set(args.methods)
        unknown = enabled_methods - ALL_METHOD_KEYS
        if unknown:
            parser.error(f"Unknown methods: {unknown}. Choose from {sorted(ALL_METHOD_KEYS)}")

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    device = args.device
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n_train_list = args.n_train

    # Resolve n_jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = max(1, cpu_count() + n_jobs)
    n_jobs = min(n_jobs, len(seeds))

    print(f"{'=' * 70}")
    print(f"  Repeated Experiment Runner (parallel)")
    print(f"  Experiments: {args.expname}")
    print(f"  n_train: {n_train_list}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({args.n_seeds} total)")
    print(f"  Methods: {sorted(enabled_methods)}")
    print(f"  Device: {device}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")

    for expname in args.expname:
        exp_cfg = EXPERIMENTS[expname]
        desc = exp_cfg.get('description', '')

        for n_train in n_train_list:
            print(f"\n{'#' * 70}")
            print(f"  {expname}: n_var={exp_cfg['n_var']}, n_train={n_train}, "
                  f"noise_std={exp_cfg['noise_std']}")
            if desc:
                print(f"  {desc}")
            print(f"{'#' * 70}")

            # Per-seed log file: append after each seed so we can resume
            log_path = os.path.join(out_dir, f'{expname}_n{n_train}_log.csv')

            # Map enabled method keys -> display names for CSV filtering
            enabled_display_names = set()
            for key in enabled_methods:
                if key in METHOD_RUNNERS:
                    enabled_display_names.add(METHOD_RUNNERS[key][0])

            # --override: remove existing rows for the enabled methods
            if args.override and os.path.exists(log_path):
                lock_path = log_path + '.lock'
                lock = filelock.FileLock(lock_path, timeout=60)
                with lock:
                    try:
                        df_existing = pd.read_csv(log_path)
                        n_before = len(df_existing)
                        df_existing = df_existing[~df_existing['method'].isin(enabled_display_names)]
                        n_removed = n_before - len(df_existing)
                        if n_removed > 0:
                            if len(df_existing) > 0:
                                df_existing.to_csv(log_path, index=False)
                            else:
                                os.remove(log_path)
                            print(f"  Override: removed {n_removed} rows for {sorted(enabled_display_names)}")
                        else:
                            print(f"  Override: no existing rows to remove for {sorted(enabled_display_names)}")
                    except Exception:
                        pass

            # Check for existing completed (seed, method) pairs (resume support)
            completed_seeds = set()
            # Per-seed set of already-completed method display names
            seed_completed_methods = {}  # seed -> set of display names
            if os.path.exists(log_path):
                try:
                    df_existing = pd.read_csv(log_path)
                    for s in df_existing['seed'].unique():
                        done = set(df_existing[df_existing['seed'] == s]['method'].values)
                        seed_completed_methods[s] = done
                        if enabled_display_names.issubset(done):
                            completed_seeds.add(s)
                    n_partial = sum(1 for s in seeds
                                    if s in seed_completed_methods
                                    and s not in completed_seeds
                                    and seed_completed_methods[s] & enabled_display_names)
                    if completed_seeds:
                        print(f"  Resuming: {len(completed_seeds)} fully-completed seeds, "
                              f"{n_partial} partially-completed")
                except Exception:
                    seed_completed_methods = {}
                    completed_seeds = set()

            # Filter out already-completed seeds
            pending_seeds = [s for s in seeds if s not in completed_seeds]
            if not pending_seeds:
                print(f"  All {len(seeds)} seeds already completed. Skipping.")
            elif n_jobs == 1:
                # Sequential execution (original behavior)
                for i, seed in enumerate(pending_seeds):
                    t0 = time.time()
                    idx = seeds.index(seed)

                    # Determine which methods still need to run for this seed
                    already_done = seed_completed_methods.get(seed, set())
                    seed_enabled = set()
                    for key in enabled_methods:
                        if key in METHOD_RUNNERS:
                            dname = METHOD_RUNNERS[key][0]
                        else:
                            continue
                        if dname not in already_done:
                            seed_enabled.add(key)

                    if not seed_enabled:
                        continue
                    skipped = enabled_methods - seed_enabled
                    skip_msg = f" (skipping already-done: {sorted(skipped)})" if skipped else ""
                    print(f"\n  Seed {seed} ({idx+1}/{args.n_seeds}){skip_msg} ...", end='', flush=True)

                    methods = run_single_seed(expname, exp_cfg, n_train, seed, device,
                                              seed_enabled)

                    # Write results for this seed immediately to log
                    seed_rows = []
                    for mname, res in methods.items():
                        seed_rows.append({
                            'seed': seed,
                            'method': mname,
                            'rmse': res.get('rmse', np.nan),
                            'y_width_at_95': res.get('y_width_at_95', np.nan),
                        })

                    _append_to_log(log_path, seed_rows)

                    elapsed = time.time() - t0
                    print(f" done ({elapsed:.1f}s)")
            else:
                # Parallel execution -- per-seed method filtering
                print(f"  Running {len(pending_seeds)} seeds with {n_jobs} workers...")
                worker_args = []
                for seed in pending_seeds:
                    already_done = seed_completed_methods.get(seed, set())
                    seed_enabled = set()
                    for key in enabled_methods:
                        if key in METHOD_RUNNERS:
                            dname = METHOD_RUNNERS[key][0]
                        else:
                            continue
                        if dname not in already_done:
                            seed_enabled.add(key)
                    if seed_enabled:
                        worker_args.append(
                            (expname, expname, n_train, seed, device, seed_enabled)
                        )

                t_start = time.time()
                with Pool(processes=n_jobs) as pool:
                    results = pool.map(_worker_fn, worker_args)

                # Write all results to log (in seed order)
                for seed, seed_rows, elapsed in sorted(results, key=lambda x: x[0]):
                    if seed_rows:
                        _append_to_log(log_path, seed_rows)
                    print(f"    Seed {seed}: {len(seed_rows)} results ({elapsed:.1f}s)")

                total_time = time.time() - t_start
                print(f"  Parallel block done in {total_time:.1f}s "
                      f"(avg {total_time/len(pending_seeds):.1f}s/seed)")

            # --- Aggregate from log file ---
            if not os.path.exists(log_path):
                print("  No results to aggregate.")
                continue

            print(f"\n{'=' * 70}")
            print(f"  RESULTS: {expname}, n_train={n_train}")
            print(f"{'=' * 70}")

            df_all = pd.read_csv(log_path)
            n_seeds_done = df_all['seed'].nunique()
            all_methods = sorted(df_all['method'].unique())

            # Build summary DataFrame
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

            # Print table
            print(f"\n{'Method':<26} {'RMSE':>16} {'y-W@95':>16} {'N':>3}")
            print('-' * 65)
            for _, row in df_summary.iterrows():
                rmse_str = f"{row['RMSE (mean)']:.4f}+/-{row['RMSE (std)']:.4f}"
                yw95_str = f"{row['y-W@95 (mean)']:.4f}+/-{row['y-W@95 (std)']:.4f}" if not np.isnan(row.get('y-W@95 (mean)', np.nan)) else "N/A"
                print(f"{row['Method']:<26} {rmse_str:>16} {yw95_str:>16} {row['N runs']:>3.0f}")

            # Save summary CSV
            summary_path = os.path.join(out_dir, f'{expname}_n{n_train}_summary.csv')
            df_summary.to_csv(summary_path, index=False)
            print(f"\n  Summary: {summary_path}")
            print(f"  Log:     {log_path} ({n_seeds_done} seeds)")

    print(f"\nDone. All results in {out_dir}/")


if __name__ == '__main__':
    main()
