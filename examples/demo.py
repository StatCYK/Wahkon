#!/usr/bin/env python3
"""
demo.py — Quick-start example for the Wahkon (ProfileWKN) network.

This script demonstrates:
  1. Creating a synthetic regression dataset
  2. Selecting regularization hyperparameters via Bayesian optimization
  3. Training a ProfileWKN model
  4. Point prediction and RMSE evaluation

Target function (f₁ from the paper):
    f(x) = log(x₁² + x₂² + |tan(x₃)|)
           + cot(π / (1 + exp(x₁² + sin(6x₂) + x₃²)))  + noise

Usage
-----
    cd wahkon/
    pip install -e .
    python examples/demo.py
    python examples/demo.py --n_train 400 --seed 0
    python examples/demo.py --expname exp1 --n_train 200
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wahkon import ProfileWKN, create_dataset, select_lambda_twostage


# ============================================================
#  Experiment definitions
# ============================================================
EXPERIMENTS = {
    'exp1': {
        'name': 'f₁ (D=3)',
        'f': lambda x: (
            torch.log(x[:, [0]] ** 2 + x[:, [1]] ** 2 + abs(torch.tan(x[:, [2]])))
            + 1 / torch.tan(torch.pi / (1 + torch.exp(
                x[:, [0]] ** 2 + torch.sin(6 * x[:, [1]]) + x[:, [2]] ** 2)))
            + 0.950702 * torch.randn(x.shape[0], 1)
        ),
        'f_true': lambda x: (
            torch.log(x[:, [0]] ** 2 + x[:, [1]] ** 2 + abs(torch.tan(x[:, [2]])))
            + 1 / torch.tan(torch.pi / (1 + torch.exp(
                x[:, [0]] ** 2 + torch.sin(6 * x[:, [1]]) + x[:, [2]] ** 2)))
        ),
        'width': [3, 5, 1],
        'n_var': 3,
    },
    'exp2': {
        'name': 'f₂ (D=2)',
        'f': lambda x: (
            torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
            + 0.969773 * torch.randn(x.shape[0], 1)
        ),
        'f_true': lambda x: torch.exp(
            torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2
        ),
        'width': [2, 5, 1],
        'n_var': 2,
    },
    'exp3': {
        'name': 'f₃ (D=4)',
        'f': lambda x: (
            torch.exp(0.5 * (
                torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2))
                + torch.sin(torch.pi * (x[:, [2]] ** 2 + x[:, [3]] ** 2))))
            + 0.420433 * torch.randn(x.shape[0], 1)
        ),
        'f_true': lambda x: torch.exp(0.5 * (
            torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2))
            + torch.sin(torch.pi * (x[:, [2]] ** 2 + x[:, [3]] ** 2)))),
        'width': [4, 4, 2, 1],
        'n_var': 4,
    },
}


# ============================================================
#  Helpers
# ============================================================

def num_link_fun(width):
    """Total number of univariate link functions in the network."""
    return sum(width[l] * width[l + 1] for l in range(len(width) - 1))


def lamb_scale(n_train, width):
    """Default lower-layer penalty scaling: n^{-4/5} × #links."""
    return n_train ** (-4 / 5) * num_link_fun(width)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wahkon (ProfileWKN) demo on synthetic benchmarks')
    parser.add_argument('--expname', type=str, default='exp1',
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment name (default: exp1)')
    parser.add_argument('--n_train', type=int, default=200,
                        help='Number of training samples (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Max training steps (default: 500)')
    parser.add_argument('--grid', type=int, default=9,
                        help='Number of kernel grid points (default: 9)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Gaussian kernel bandwidth (default: 0.5)')
    parser.add_argument('--skip_bo', action='store_true',
                        help='Skip BO hyperparameter search (use defaults)')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help='Output directory for plots (default: ./results)')
    args = parser.parse_args()

    exp = EXPERIMENTS[args.expname]
    width = exp['width']
    n_var = exp['n_var']
    n_train = args.n_train
    device = args.device

    print("=" * 60)
    print(f"  Wahkon Demo — {exp['name']}")
    print("=" * 60)
    print(f"  width     = {width}")
    print(f"  n_train   = {n_train}")
    print(f"  grid      = {args.grid}")
    print(f"  sigma     = {args.sigma}")
    print(f"  device    = {device}")
    print(f"  seed      = {args.seed}")
    print("=" * 60)

    # ── Step 1: Create dataset ──────────────────────────────────
    print("\n[1/5] Creating dataset...")
    dataset = create_dataset(
        exp['f'], f_true=exp['f_true'], n_var=n_var,
        ranges=(-1, 1), train_num=n_train, test_num=1000,
        normalize_input=True, normalize_label=True,
        seed=args.seed, device=device,
    )
    print(f"  Train: {dataset['train_input'].shape}")
    print(f"  Test:  {dataset['test_input'].shape}")

    # ── Step 2: Hyperparameter selection ────────────────────────
    scale = lamb_scale(n_train, width)

    if args.skip_bo:
        best_lamb_lower = scale              # fixed by formula
        best_lamb_last = scale * 0.5         # reasonable default
        print(f"\n[2/5] Using default lambdas (--skip_bo)")
    else:
        print(f"\n[2/5] Selecting lamb_last via Bayesian optimization...")
        print(f"  Lambda scale factor: {scale:.6f}")
        print(f"  lamb_lower fixed = {scale:.6f}")
        best_lamb_last, best_lamb_lower = select_lambda_twostage(
            width=width, dataset=dataset,
            n_splits=5, steps=args.steps, lr=0.005,
            grid=args.grid, sigma=args.sigma,
            n_calls=15, n_random_init=5,
            lamb_last_range=(0.01, 3.0),
            batch=200,
            device=device, random_state=args.seed,
            seed=args.seed,
        )

    print(f"  lamb_lower = {best_lamb_lower:.6f}")
    print(f"  lamb_last  = {best_lamb_last:.6f}")

    # ── Step 3: Train ───────────────────────────────────────────
    print(f"\n[3/4] Training ProfileWKN ({args.steps} steps max)...")
    model = ProfileWKN(
        width=width, grid=args.grid, sigma=args.sigma,
        seed=args.seed, device=device,
    )
    results, _, _ = model.train(
        dataset, opt='Adam', steps=args.steps, lr=0.005,
        lamb_last=best_lamb_last, lamb_lower=best_lamb_lower,
        batch=200, update_grid=False,
        verbose=True, device=device,
        early_stopping=True, patience=50, min_delta=1e-5,
    )

    # Refit last layer on full training set
    model.fit_last_layer(
        dataset['train_input'], dataset['train_label'],
        lamb_last=best_lamb_last, device=device,
    )

    # ── Step 4: Evaluate ────────────────────────────────────────
    print(f"\n[4/4] Evaluating...")
    with torch.no_grad():
        f_hat = model.predict(
            dataset['train_input'], dataset['test_input'], device=device,
        )
        y_true = dataset['test_true'].view(-1).to(device)
        test_rmse = float(torch.sqrt(torch.mean((y_true - f_hat) ** 2)))

    print(f"  Test RMSE (vs true function): {test_rmse:.4f}")

    # ── Save plot ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Training curves
    axes[0].plot(results['train_loss'], label='Train RMSE', alpha=0.8)
    axes[0].plot(results['test_loss'], label='Test RMSE', alpha=0.8)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)

    # Panel 2: Profile objective
    axes[1].plot(results['profile_obj'], color='tab:red', alpha=0.8)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Profile Objective')
    axes[1].set_title('Profile Objective')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)

    plt.suptitle(
        f'{exp["name"]}  |  n={n_train}  |  RMSE={test_rmse:.4f}',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f'{args.expname}_n{n_train}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {fig_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
