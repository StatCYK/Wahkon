#!/usr/bin/env python
"""
plot_results.py -- Plot log-RMSE vs n_train for different methods with error bars.

Reads the per-seed log CSVs produced by run_repeated.py and generates one
line plot per experiment showing log10(RMSE) on the y-axis, n_train on the
x-axis, with each method as a separate line +/- 1-std error bars.

Usage
-----
    # Plot all experiments found in the results directory:
    python plot_results.py --results_dir results/simu --expname f1 f2 f3 f4 \
        --methods Wahkon KAN MLP NTK BNN --combined \
        --n_train 200 400 800 1600 3200 10000

    # Plot specific experiments:
    python plot_results.py --results_dir results/run1 --expname f1 f3

    # Plot only specific methods:
    python plot_results.py --results_dir results/run1 --methods Wahkon KAN MLP

    # Customize figure size and font:
    python plot_results.py --results_dir results/run1 --figsize 8 5 --fontsize 12
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

# -- Experiment-name mapping ---------------------------------------------------
# Internal CSV name -> (display label, dimension)
EXPNAME_MAP = {
    'f1': ('$f_1$', 3),
    'f2': ('$f_2$', 10),
    'f3': ('$f_3$', 4),
    'f4': ('$f_4$', 6),
    # Backward-compat aliases for old CSV filenames
    'exp5':        ('$f_1$', 3),
    'sin_sum_sq':  ('$f_2$', 10),
    'exp3':        ('$f_3$', 4),
    'nested_comp': ('$f_4$', 6),
}

# Canonical order for the combined panel figure
EXPNAME_ORDER = ['f1', 'f2', 'f3', 'f4']

def _display_name(expname):
    """Return a formatted title string for a given experiment."""
    if expname in EXPNAME_MAP:
        label, dim = EXPNAME_MAP[expname]
        return f'{label} ($D={dim}$)'
    return expname

# -- Method style --------------------------------------------------------------
METHOD_STYLE = {
    'Wahkon':      {'color': 'red',     'marker': 'o', 'ls': '-'},
    'KAN':         {'color': '#2CA02C', 'marker': '^', 'ls': '--'},
    'MLP':         {'color': '#1F77B4', 'marker': 'D', 'ls': '--'},
    'NTK':         {'color': '#4A86C8', 'marker': '>', 'ls': ':'},
    'BNN':         {'color': '#9467BD', 'marker': 'h', 'ls': '-.'},
    'MLP (Deep)':  {'color': '#FF7F0E', 'marker': 's', 'ls': '--'},
}

# Display-name overrides: CSV method name -> legend label
METHOD_DISPLAY = {
    'Wahkon':        'Wahkon',
    'NTK':           'NTK',
    'Profile WKN':   'Wahkon',     # backward compat
    'NTK (NT-BO)':   'NTK',        # backward compat
    'KAN (pykan)':   'KAN',         # backward compat
}

def _method_label(method):
    """Return the legend label for a method (applies display-name overrides)."""
    return METHOD_DISPLAY.get(method, method)

# Default method subset matching the manuscript comparisons
DEFAULT_METHODS = ['Wahkon', 'KAN', 'MLP', 'NTK', 'BNN', 'MLP (Deep)']

# Fallback palette for unknown method names
_FALLBACK_COLORS = ['#393B79', '#637939', '#8C6D31', '#843C39',
                    '#7B4173', '#5254A3', '#6B6ECF', '#9C9EDE']
_FALLBACK_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X']


def _get_style(method_name, idx):
    """Return (color, marker, linestyle) for a method."""
    if method_name in METHOD_STYLE:
        s = METHOD_STYLE[method_name]
        return s['color'], s['marker'], s['ls']
    i = idx % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[i], _FALLBACK_MARKERS[i], '-'


def _method_order_key(method):
    """Sort key so default methods appear first in legend order."""
    order = list(METHOD_STYLE.keys())
    try:
        return order.index(method)
    except ValueError:
        return len(order)


def discover_experiments(results_dir):
    """Scan results_dir for log CSVs and return {expname: {n_train: csv_path}}."""
    pattern = os.path.join(results_dir, '*_n*_log.csv')
    files = glob.glob(pattern)

    experiments = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(r'^(.+)_n(\d+)_log\.csv$', fname)
        if not m:
            continue
        expname, n_train = m.group(1), int(m.group(2))
        experiments.setdefault(expname, {})[n_train] = fpath

    return experiments


def load_experiment(csv_dict):
    """Load and concatenate all n_train CSVs for one experiment.
    Returns a DataFrame with an added 'n_train' column.
    """
    frames = []
    for n_train, path in sorted(csv_dict.items()):
        df = pd.read_csv(path)
        df['n_train'] = n_train
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _plot_on_ax(ax, df, methods_filter, fontsize, title=None):
    """Core plotting logic: draw log10(RMSE) vs n_train on a given Axes."""
    all_methods = sorted(df['method'].unique(), key=_method_order_key)
    if methods_filter:
        all_methods = [m for m in all_methods if m in methods_filter]

    n_trains = sorted(df['n_train'].unique())

    fallback_idx = 0
    for method in all_methods:
        mdf = df[df['method'] == method]

        means, stds, xs = [], [], []
        for nt in n_trains:
            vals = mdf[mdf['n_train'] == nt]['rmse'].dropna()
            if len(vals) == 0:
                continue
            log_vals = np.log10(vals.values)
            means.append(log_vals.mean())
            stds.append(log_vals.std())
            xs.append(nt)

        if not xs:
            continue

        color, marker, ls = _get_style(method, fallback_idx)
        if method not in METHOD_STYLE:
            fallback_idx += 1

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(xs, means, yerr=stds,
                    label=_method_label(method), color=color, marker=marker,
                    linestyle=ls, linewidth=1.8, markersize=6,
                    capsize=3, capthick=1.2, elinewidth=1.0)

    ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=fontsize)
    ax.set_ylabel('$\\log_{10}(\\mathrm{RMSE})$', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, loc='left')

    # Log-scale x-axis if n_trains span a wide range
    if len(n_trains) > 1 and max(n_trains) / min(n_trains) >= 4:
        ax.set_xscale('log', base=2)
        ax.set_xticks(n_trains)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.3)


def plot_log_rmse(df, expname, out_path, methods_filter=None,
                  figsize=(7, 4.5), fontsize=11):
    """
    Draw log10(RMSE) vs n_train for each method, with +/-1-std error bars.
    Single-panel figure for one experiment.
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = _display_name(expname)
    _plot_on_ax(ax, df, methods_filter, fontsize, title=title)
    ax.legend(fontsize=fontsize - 2, loc='best', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined(experiments_data, out_path, methods_filter=None,
                  figsize=None, fontsize=11):
    """
    Draw a 2x2 combined panel figure for experiments f1-f4.

    Parameters
    ----------
    experiments_data : dict  {expname: DataFrame}
        Must contain entries keyed by the internal experiment names.
    """
    # Determine which experiments to show in canonical order
    exp_keys = [k for k in EXPNAME_ORDER if k in experiments_data]
    n_exp = len(exp_keys)
    if n_exp == 0:
        print("  No matching experiments for combined plot.")
        return

    ncols = n_exp
    nrows = 1
    if figsize is None:
        figsize = (3.5 * ncols, 3.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, expname in enumerate(exp_keys):
        ax = axes[0][idx]
        df = experiments_data[expname]
        panel_label = chr(ord('a') + idx)  # (a), (b), ...
        title = f'({panel_label}) {_display_name(expname)}'
        _plot_on_ax(ax, df, methods_filter, fontsize, title=title)
        # Only show y-label on the leftmost panel
        if idx > 0:
            ax.set_ylabel('')

    # Hide unused axes
    for idx in range(n_exp, ncols):
        axes[0][idx].set_visible(False)

    # Shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), fontsize=fontsize - 1,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined panel: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot log-RMSE vs n_train from repeated-experiment logs',
    )
    parser.add_argument('--results_dir', type=str, default='./results/simu',
                        help='Directory containing *_n*_log.csv files (default: ./results/simu)')
    parser.add_argument('--expname', type=str, nargs='+',
                        default=['f1', 'f2', 'f3', 'f4'],
                        help='Experiment names to plot (default: f1 f2 f3 f4)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to include (display names, e.g. Wahkon KAN MLP NTK). '
                             'Default: Wahkon, KAN, MLP, NTK, BNN, MLP (Deep).')
    parser.add_argument('--all_methods', action='store_true',
                        help='Plot all methods found in the CSVs (overrides --methods)')
    parser.add_argument('--combined', action='store_true',
                        help='Generate a combined 2x2 panel figure for f1-f4')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 4.5],
                        help='Figure size in inches: width height (default: 7 4.5)')
    parser.add_argument('--fontsize', type=int, default=11,
                        help='Base font size (default: 11)')
    parser.add_argument('--n_train', type=int, nargs='+', default=None,
                        help='Training sizes to plot (default: all found)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory to save plots (default: same as --results_dir)')
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory")
        return

    experiments = discover_experiments(results_dir)
    if not experiments:
        print(f"No log CSVs found in {results_dir}")
        return

    # Filter to requested experiments
    if args.expname:
        experiments = {k: v for k, v in experiments.items() if k in args.expname}
        missing = set(args.expname) - set(experiments.keys())
        if missing:
            print(f"Warning: no CSVs found for experiments: {missing}")

    os.makedirs(out_dir, exist_ok=True)

    # Resolve method filter
    if args.all_methods:
        methods_filter = None
    elif args.methods is not None:
        methods_filter = args.methods
    else:
        methods_filter = DEFAULT_METHODS

    print(f"Found {len(experiments)} experiment(s): {sorted(experiments.keys())}")
    if methods_filter:
        print(f"Method filter: {methods_filter}")
    if args.n_train:
        print(f"n_train filter: {args.n_train}")

    # Load all experiment data
    experiments_data = {}
    for expname in sorted(experiments.keys()):
        csv_dict = experiments[expname]

        # Filter to requested n_train values
        if args.n_train is not None:
            csv_dict = {nt: p for nt, p in csv_dict.items()
                        if nt in args.n_train}
            if not csv_dict:
                print(f"\n  {expname}: no data for n_train={args.n_train}, skipping")
                continue

        n_trains = sorted(csv_dict.keys())
        print(f"\n  {expname} -> {_display_name(expname)}: n_train = {n_trains}")

        df = load_experiment(csv_dict)
        all_methods = sorted(df['method'].unique())
        print(f"  Methods in data: {all_methods}")
        print(f"  Seeds: {df['seed'].nunique()}")

        experiments_data[expname] = df

        # Individual plot
        out_path = os.path.join(out_dir, f'{expname}_log_rmse.png')
        plot_log_rmse(df, expname, out_path,
                      methods_filter=methods_filter,
                      figsize=tuple(args.figsize),
                      fontsize=args.fontsize)

    # Combined panel plot
    if args.combined:
        combined_path = os.path.join(out_dir, 'combined_log_rmse.png')
        plot_combined(experiments_data, combined_path,
                      methods_filter=methods_filter,
                      fontsize=args.fontsize)

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == '__main__':
    main()
