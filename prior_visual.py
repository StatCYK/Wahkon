#!/usr/bin/env python3
"""
prior_visual.py — Monte Carlo simulation of the Wahkon prior.

Draws nrep samples from the GP prior at each of L layers (additive RBF
kernel), then produces:
  1. A Q-Q plot of squared Mahalanobis distances vs chi-squared quantiles
     (checking joint normality of the hidden-layer outputs).
  2. An empirical CDF plot of the squared Mahalanobis distances.

Usage
-----
    python prior_visual.py
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2

_here = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ────────────────────────────────────────────
train_num = 100
n_var = 4
L = 5               # number of layers
width = n_var
sigma = 1
nrep = 1000          # Monte Carlo repetitions
seed = 12345

# Reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── Generate training inputs ─────────────────────────────────
ranges = np.array([-1, 1] * n_var).reshape(n_var, 2)

train_input = torch.zeros(train_num, n_var)
for i in range(n_var):
    train_input[:, i] = (torch.rand(train_num) *
                         (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0])

# ── Monte Carlo: sample from the GP prior at each layer ──────
X_all_layers_MC = []
for rep in range(nrep):
    input_l = train_input
    X_all_layers = []
    for l in range(L):
        Cov_l = sum(
            torch.exp(-0.5 * (input_l[:, k].unsqueeze(1) -
                              input_l[:, k].unsqueeze(0)) ** 2 / sigma ** 2)
            for k in range(width)
        ) / width
        Cov_l += 1e-4 * torch.eye(train_num)
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(train_num), covariance_matrix=Cov_l)
        x_l = mvn.sample((width,)).T
        X_all_layers.append(x_l)
        input_l = x_l
    X_all_layers_MC.append(X_all_layers)


# ── Helper ───────────────────────────────────────────────────
def mahalanobis_distance(data):
    """Compute squared Mahalanobis distance for each row."""
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    centered_data = data - mean
    md = np.sum(centered_data @ inv_cov * centered_data, axis=1)
    return md


# ── Colours ──────────────────────────────────────────────────
colors = plt.cm.viridis(np.linspace(1, 0, L))

# Create output directory
fig_dir = os.path.join(_here, 'figure')
os.makedirs(fig_dir, exist_ok=True)

# ── Plot 1: Q-Q plot (Mahalanobis vs chi-squared) ────────────
plt.figure(figsize=(8, 6))

for l in range(L):
    data = np.array([rep[l][:, 0].numpy() for rep in X_all_layers_MC])
    md = mahalanobis_distance(data)
    chi2_quantiles = chi2.ppf(np.linspace(0.01, 0.99, len(md)), df=train_num)
    md_sorted = np.sort(md)
    chi2_quantiles_sorted = np.sort(chi2_quantiles)
    plt.scatter(md_sorted, chi2_quantiles_sorted, s=10,
                label=f'Layer {l+1}', color=colors[l], alpha=0.7)

x = np.linspace(chi2.ppf(0.01, df=train_num),
                chi2.ppf(0.99, df=train_num), 500)
plt.plot(x, x, '-', label='Ideal Line', linewidth=2, color='black')

plt.title("Q-Q Plot for checking the normality of the "
          "hidden layer's outputs", fontsize=16)
plt.xlabel("Squared Mahalanobis Distance", fontsize=14)
plt.ylabel("Chi-squared Quantiles", fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"width_{width}_overlaid_qq_plots.png"),
            dpi=300)
print(f"Saved Q-Q plot to {fig_dir}/width_{width}_overlaid_qq_plots.png")

# ── Plot 2: Empirical CDF of squared Mahalanobis distances ──
plt.figure(figsize=(6, 6))

for l in range(L):
    data = np.array([rep[l][:, 0].numpy() for rep in X_all_layers_MC])
    md = mahalanobis_distance(data)
    sorted_md = np.sort(md)
    cdf = np.arange(1, len(sorted_md) + 1) / len(sorted_md)
    plt.plot(sorted_md, cdf, label=f'Layer {l+1}',
             color=colors[l], alpha=1, linewidth=3)

plt.ylabel("Empirical CDF", fontsize=14)
plt.xlabel("Squared Mahalanobis Distance", fontsize=14)
plt.title("Empirical CDF of the squared Mahalanobis Distances", fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"width_{width}_prior_md_hist.png"),
            dpi=300)
print(f"Saved CDF plot to {fig_dir}/width_{width}_prior_md_hist.png")
