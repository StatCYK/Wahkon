#!/usr/bin/env python3
"""
plot_ci_combined.py — Plot CI width and coverage from a single results
directory containing {expname}_n{n_train}_log.csv files.

All methods (Wahkon, NTK, BNN) write their results to the same CSV format
via run_repeated.py (Wahkon/BNN) and run_ntk_nt_bo.py (NTK).
CSV columns: seed, method, rmse, y_width_at_95  (and optionally y_coverage).

Usage
-----
    python plot_ci_combined.py \\
        --results_dir ./results \\
        --expname f1 f2 f3 f4

    python plot_ci_combined.py \\
        --results_dir ./results \\
        --methods Wahkon NTK BNN \\
        --n_train 100 200 400 800 1600 3200 10000
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ============================================================
#  Style (matching plot_results.py exactly)
# ============================================================

EXPNAME_MAP = {
    'f1': ('$f_1$', 3),
    'f2': ('$f_2$', 10),
    'f3': ('$f_3$', 4),
    'f4': ('$f_4$', 6),
    # backward compat
    'exp5':        ('$f_1$', 3),
    'sin_sum_sq':  ('$f_2$', 10),
    'exp3':        ('$f_3$', 4),
    'nested_comp': ('$f_4$', 6),
}
EXPNAME_ORDER = ['f1', 'f2', 'f3', 'f4']

def _display_name(expname):
    if expname in EXPNAME_MAP:
        label, dim = EXPNAME_MAP[expname]
        return f'{label} ($D={dim}$)'
    return expname

# Keyed by the unified method name used in the merged DataFrame
METHOD_STYLE = {
    'Wahkon':  {'color': 'red',     'marker': 'o', 'ls': '-'},
    'NTK':     {'color': '#4A86C8', 'marker': '>', 'ls': ':'},
    'BNN':     {'color': '#9467BD', 'marker': 'h', 'ls': '-.'},
}

# CSV method name → display label (legend)
METHOD_DISPLAY = {
    'Wahkon': 'Wahkon',
    'NTK':    'NTK',
    'BNN':    'BNN',
}

# Plot order
METHODS_ORDER = ['Wahkon', 'NTK', 'BNN']

# Fallback palette for unknown methods
_FALLBACK_COLORS = ['#393B79', '#637939', '#8C6D31', '#843C39',
                    '#7B4173', '#5254A3', '#6B6ECF', '#9C9EDE']
_FALLBACK_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X']

# Backward-compat aliases for method names that may appear in older CSVs
_METHOD_ALIASES = {
    'WKN (profile)': 'Wahkon',
    'Profile WKN':   'Wahkon',
}

def _method_label(method):
    return METHOD_DISPLAY.get(method, method)

def _get_style(method_name, idx=0):
    if method_name in METHOD_STYLE:
        s = METHOD_STYLE[method_name]
        return s['color'], s['marker'], s['ls']
    i = idx % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[i], _FALLBACK_MARKERS[i], '-'

def _method_order_key(method):
    try:
        return METHODS_ORDER.index(method)
    except ValueError:
        return len(METHODS_ORDER)


# ============================================================
#  Data loading
# ============================================================

def load_results(results_dir, expnames, n_trains, methods):
    """
    Load results from {expname}_n{n_train}_log.csv files.

    CSV columns: seed, method, rmse, y_width_at_95  (and optionally
    y_coverage, f_coverage, etc.).

    Parameters
    ----------
    results_dir : str
        Directory containing the CSV files.
    expnames : list[str]
        Experiment names to look for (e.g. ['f1', 'f2', 'f3', 'f4']).
    n_trains : list[int]
        Training sizes to look for.
    methods : list[str]
        Method names to keep (after alias resolution).

    Returns
    -------
    pd.DataFrame with columns:
        expname, n_train, seed, method, rmse, y_width_at_95,
        and optionally y_coverage, f_coverage.
    """
    frames = []
    # Also check backward-compat expname aliases
    alias_to_canonical = {}
    for alias, (label, dim) in EXPNAME_MAP.items():
        # Find the canonical name (f1-f4) that maps to this (label, dim)
        for canon in EXPNAME_ORDER:
            if EXPNAME_MAP.get(canon) == (label, dim):
                alias_to_canonical[alias] = canon
                break

    # Build set of expnames to search for (canonical + aliases)
    search_expnames = set(expnames)
    for alias, canon in alias_to_canonical.items():
        if canon in expnames:
            search_expnames.add(alias)

    # Discover all matching CSV files
    pattern = os.path.join(results_dir, '*_n*_log.csv')
    csv_files = glob.glob(pattern)

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        m = re.match(r'^(.+)_n(\d+)_log\.csv$', fname)
        if not m:
            continue
        file_expname = m.group(1)
        file_n_train = int(m.group(2))

        # Check if this expname is one we want
        if file_expname not in search_expnames:
            continue

        # Check if this n_train is one we want
        if n_trains and file_n_train not in n_trains:
            continue

        # Map to canonical expname
        canonical_expname = alias_to_canonical.get(file_expname, file_expname)

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")
            continue

        if len(df) == 0:
            continue

        # Apply method name aliases for backward compat
        if 'method' in df.columns:
            df['method'] = df['method'].replace(_METHOD_ALIASES)

        # Filter to requested methods
        if methods:
            df = df[df['method'].isin(methods)]
        if len(df) == 0:
            continue

        # Build output DataFrame with unified columns
        out = pd.DataFrame({
            'expname': canonical_expname,
            'n_train': file_n_train,
            'seed': df['seed'].values,
            'method': df['method'].values,
            'rmse': df['rmse'].values if 'rmse' in df.columns else np.nan,
            'y_width_at_95': df['y_width_at_95'].values if 'y_width_at_95' in df.columns else np.nan,
        })

        # Optional columns
        if 'y_coverage' in df.columns:
            out['y_coverage'] = df['y_coverage'].values
        if 'f_coverage' in df.columns:
            out['f_coverage'] = df['f_coverage'].values

        frames.append(out)

    if not frames:
        print(f"  No matching CSV files found in {results_dir}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ============================================================
#  Plotting
# ============================================================

def _plot_on_ax(ax, df, metric, ylabel, fontsize, title=None,
                hline=None, hline_label=None):
    """Core plotting logic: draw metric vs n_train on a given Axes.
    Matches plot_results.py style exactly."""
    all_methods = sorted(df['method'].unique(), key=_method_order_key)
    n_trains = sorted(df['n_train'].unique())

    fallback_idx = 0
    for method in all_methods:
        mdf = df[df['method'] == method]
        means, stds, xs = [], [], []
        for nt in n_trains:
            vals = mdf[mdf['n_train'] == nt][metric].dropna().values
            if len(vals) == 0:
                continue
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            xs.append(nt)
        if not xs:
            continue

        color, marker, ls = _get_style(method, fallback_idx)
        if method not in METHOD_STYLE:
            fallback_idx += 1

        ax.errorbar(xs, np.array(means), yerr=np.array(stds),
                    label=_method_label(method), color=color, marker=marker,
                    linestyle=ls, linewidth=1.8, markersize=6,
                    capsize=3, capthick=1.2, elinewidth=1.0)

    if hline is not None:
        ax.axhline(y=hline, color='grey', ls='--', alpha=0.6,
                   label=hline_label or '')

    ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, loc='left')

    if len(n_trains) > 1 and max(n_trains) / min(n_trains) >= 4:
        ax.set_xscale('log', base=2)
        ax.set_xticks(n_trains)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.3)


def plot_combined_grid(df, out_path, exps_order=None, fontsize=11):
    """
    Combined panel figure: one row per experiment, two columns
    (y-Width@95 and y-Coverage).  If y_coverage is not present in the
    data, only a single column (y-Width@95) is produced.
    """
    if exps_order is None:
        exps_order = sorted(df['expname'].unique())
    exps = [e for e in exps_order if e in df['expname'].values]
    if not exps:
        exps = sorted(df['expname'].unique())
    n_exp = len(exps)
    if n_exp == 0:
        return

    has_coverage = ('y_coverage' in df.columns and
                    df['y_coverage'].notna().any())
    n_cols = 2 if has_coverage else 1
    figsize = (5.0 * n_cols, 3.2 * n_exp)

    fig, axes = plt.subplots(n_exp, n_cols, figsize=figsize, squeeze=False)

    for row, expname in enumerate(exps):
        edf = df[df['expname'] == expname]
        panel_label_base = row * n_cols

        # Left column: y-Width@95
        ax_width = axes[row][0]
        label_idx = chr(ord('a') + panel_label_base)
        title = f'({label_idx}) {_display_name(expname)}'
        _plot_on_ax(ax_width, edf, 'y_width_at_95',
                    'y-Width@95', fontsize, title=title)

        # Right column: y-Coverage (if available)
        if has_coverage:
            ax_cov = axes[row][1]
            label_idx = chr(ord('a') + panel_label_base + 1)
            title = f'({label_idx}) {_display_name(expname)}'
            _plot_on_ax(ax_cov, edf, 'y_coverage',
                        'y-Coverage', fontsize, title=title,
                        hline=0.95, hline_label='95%')

    # Hide unused axes (shouldn't happen but just in case)
    for row in range(n_exp, axes.shape[0]):
        for col in range(n_cols):
            axes[row][col].set_visible(False)

    # Shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), fontsize=fontsize - 1,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_single_metric(df, metric, ylabel, out_path,
                       exps_order=None, fontsize=11,
                       hline=None, hline_label=None, figsize=None):
    """Single-row panel figure (one column per experiment) for a given metric."""
    if exps_order is None:
        exps_order = sorted(df['expname'].unique())
    exps = [e for e in exps_order if e in df['expname'].values]
    if not exps:
        exps = sorted(df['expname'].unique())
    n_exp = len(exps)
    if n_exp == 0:
        return

    if figsize is None:
        figsize = (3.5 * n_exp, 3.8)

    fig, axes = plt.subplots(1, n_exp, figsize=figsize, squeeze=False)

    for idx, expname in enumerate(exps):
        ax = axes[0][idx]
        edf = df[df['expname'] == expname]
        panel_label = chr(ord('a') + idx)
        title = f'({panel_label}) {_display_name(expname)}'
        _plot_on_ax(ax, edf, metric, ylabel, fontsize, title=title,
                    hline=hline, hline_label=hline_label)
        if idx > 0:
            ax.set_ylabel('')

    for idx in range(n_exp, len(axes[0])):
        axes[0][idx].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), fontsize=fontsize - 1,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot CI width and coverage from a single results '
                    'directory containing {expname}_n{n_train}_log.csv files')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory with *_log.csv result files '
                             '(default: ./results)')
    parser.add_argument('--expname', type=str, nargs='+',
                        default=['f1', 'f2', 'f3', 'f4'],
                        help='Experiments to plot (default: f1 f2 f3 f4)')
    parser.add_argument('--n_train', type=int, nargs='+',
                        default=[100, 200, 400, 800, 1600, 3200, 10000],
                        help='Training sizes to plot '
                             '(default: 100 200 400 800 1600 3200 10000)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['Wahkon', 'NTK', 'BNN'],
                        help='Methods to plot (default: Wahkon NTK BNN)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: --results_dir)')
    parser.add_argument('--fontsize', type=int, default=11)
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    print(f"  Experiments: {args.expname}")
    print(f"  n_train: {args.n_train}")
    print(f"  Methods: {args.methods}")

    df = load_results(args.results_dir, args.expname, args.n_train,
                      args.methods)

    if len(df) == 0:
        print("No data found. Check --results_dir path and file names.")
        return

    print(f"\nLoaded {len(df)} rows")
    print(f"  Methods found: {sorted(df['method'].unique())}")
    print(f"  Experiments: {sorted(df['expname'].unique())}")
    print(f"  n_train values: {sorted(df['n_train'].unique())}")

    # Determine experiment order
    exps = [e for e in EXPNAME_ORDER if e in df['expname'].values]
    if not exps:
        exps = sorted(df['expname'].unique())

    # Save combined CSV
    combined_path = os.path.join(out_dir, 'ci_combined_all.csv')
    df.to_csv(combined_path, index=False)
    print(f"  Saved combined CSV: {combined_path}")

    # ── Combined grid figure (main output) ────────────────────
    plot_combined_grid(
        df,
        os.path.join(out_dir, 'combined_ci_grid.png'),
        exps_order=exps, fontsize=args.fontsize)

    # ── Individual metric plots ───────────────────────────────

    # y-Width@95 vs n_train (single row)
    plot_single_metric(
        df, 'y_width_at_95', 'y-Width@95',
        os.path.join(out_dir, 'combined_y_width_vs_ntrain.png'),
        exps_order=exps, fontsize=args.fontsize)

    # y-Coverage vs n_train (if available)
    if 'y_coverage' in df.columns and df['y_coverage'].notna().any():
        plot_single_metric(
            df, 'y_coverage', 'y-Coverage',
            os.path.join(out_dir, 'combined_y_coverage_vs_ntrain.png'),
            exps_order=exps, fontsize=args.fontsize,
            hline=0.95, hline_label='95%')

    # f-Coverage vs n_train (if available)
    if 'f_coverage' in df.columns and df['f_coverage'].notna().any():
        plot_single_metric(
            df, 'f_coverage', 'f-Coverage',
            os.path.join(out_dir, 'combined_f_coverage_vs_ntrain.png'),
            exps_order=exps, fontsize=args.fontsize,
            hline=0.95, hline_label='95%')

    # RMSE vs n_train
    if 'rmse' in df.columns and df['rmse'].notna().any():
        plot_single_metric(
            df, 'rmse', 'RMSE',
            os.path.join(out_dir, 'combined_rmse_vs_ntrain.png'),
            exps_order=exps, fontsize=args.fontsize)

    # ── Summary table ─────────────────────────────────────────
    has_coverage = 'y_coverage' in df.columns and df['y_coverage'].notna().any()

    header = f"  {'Exp':<14} {'n':>5} {'Method':<10} {'RMSE':>8} {'y-Width':>9}"
    if has_coverage:
        header += f" {'y-Cov':>7}"

    print(f"\n{'=' * len(header)}")
    print(f"{'SUMMARY':^{len(header)}}")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for expname in exps:
        for nt in sorted(df[df['expname'] == expname]['n_train'].unique()):
            edf = df[(df['expname'] == expname) & (df['n_train'] == nt)]
            for m in METHODS_ORDER:
                mdf = edf[edf['method'] == m]
                if len(mdf) == 0:
                    continue
                line = (f"  {expname:<14} {nt:>5} {m:<10} "
                        f"{mdf['rmse'].mean():>8.4f} "
                        f"{mdf['y_width_at_95'].mean():>9.4f}")
                if has_coverage:
                    cov_val = mdf['y_coverage'].mean()
                    line += f" {cov_val:>7.3f}" if not np.isnan(cov_val) else f" {'N/A':>7}"
                print(line)
            print(f"  {'-' * (len(header) - 2)}")

    print(f"\nDone! All plots in {out_dir}/")


if __name__ == '__main__':
    main()
