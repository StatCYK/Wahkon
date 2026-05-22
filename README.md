# Wahkon

**A Statistically Principled Deep RKHS Superposition Network**

Wahkon unifies [Kolmogorov's superposition theorem](https://en.wikipedia.org/wiki/Kolmogorov%27s_theorem) with reproducing kernel Hilbert space (RKHS) regularization in the smoothing-spline tradition of Wahba. Each edge of the network carries a *learned univariate link function* constrained to lie in a common RKHS, yielding a framework that combines the representational power of deep architectures with the statistical rigor of kernel methods.

## Key Features

- **Deep Representer Theorem** — Despite infinite-dimensional function spaces, the estimator admits a finite kernel expansion at every layer.
- **Profile Objective** — Analytically concentrates last-layer coefficients for faster, stabler optimization.
- **Minimax-Optimal Rates** — Convergence guarantees via metric-entropy bounds under mild smoothness assumptions.

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0

### Install from source

```bash
git clone https://github.com/StatCYK/Wahkon.git
cd wahkon
pip install -e .
```

### Or install dependencies manually

```bash
# Create a virtual environment (recommended)
python -m venv wahkon_env
source wahkon_env/bin/activate  # Linux/macOS
# wahkon_env\Scripts\activate   # Windows

# Install PyTorch (visit https://pytorch.org for GPU-specific instructions)
pip install torch

# Install Wahkon
pip install -e .
```

### GPU Support

For GPU acceleration, install PyTorch with CUDA support before installing Wahkon:

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install Wahkon as above. Pass `device='cuda'` when creating datasets and training.

## Quick Start

```python
import torch
from wahkon import ProfileWKN, create_dataset

# 1. Define a target function
f_noisy = lambda x: (
    torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
    + 0.5 * torch.randn(x.shape[0], 1)
)
f_true = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

# 2. Create dataset
dataset = create_dataset(
    f_noisy, f_true=f_true, n_var=2,
    train_num=500, test_num=200, seed=42,
)

# 3. Build and train
model = ProfileWKN(width=[2, 5, 1], grid=9, sigma=0.5, device='cpu')
results, _, _ = model.train(
    dataset, opt='Adam', steps=300, lr=0.005,
    lamb_last=0.01, lamb_lower=0.01,
    batch=200, update_grid=False, device='cpu',
)

# 4. Refit last layer on full training data
model.fit_last_layer(
    dataset['train_input'], dataset['train_label'],
    lamb_last=0.01, device='cpu',
)

# 5. Predict
with torch.no_grad():
    y_hat = model.predict(
        dataset['train_input'], dataset['test_input'], device='cpu',
    )
```

## Running the Demo

```bash
# Default: exp1 (f₁, D=3), 200 training samples
python examples/demo.py

# Custom settings
python examples/demo.py --expname exp3 --n_train 400 --seed 0

# Skip Bayesian optimization (use default lambdas, runs much faster)
python examples/demo.py --skip_bo

# Use GPU
python examples/demo.py --device cuda
```

The demo trains a ProfileWKN model, evaluates point prediction RMSE, and saves a summary plot to `./results/`.

## Hyperparameter Selection

Wahkon uses a two-stage lambda selection strategy:

1. **Lower-layer penalty** (`lamb_lower`): Controls smoothness of learned feature embeddings. **Fixed** by the deterministic formula `n^{-4/5} × #links`, where `#links = Σ D_{l-1} × D_l`. This is computed internally and not tuned.
2. **Last-layer penalty** (`lamb_last`): Controls bias-variance tradeoff in the final regression. Selected via 1D Bayesian optimization over K-fold cross-validation RMSE.

```python
from wahkon import select_lambda_twostage

# lamb_lower is fixed internally; BO searches lamb_last only
best_lamb_last, fixed_lamb_lower = select_lambda_twostage(
    width=[2, 5, 1], dataset=dataset,
    n_splits=5, steps=300, lr=0.005,
    grid=9, sigma=0.5,
    n_calls=15, n_random_init=5,
    lamb_last_range=(0.01, 3.0),
    batch=200,
    device='cpu',
)
```

## Package Structure

```
wahkon/
├── pyproject.toml          # Package metadata and dependencies
├── README.md
├── LICENSE
├── wahkon/
│   ├── __init__.py         # Public API: ProfileWKN, create_dataset, ...
│   ├── profile.py          # WKN base, ProfileWKN, dataset helpers, lambda selection
│   └── core/
│       ├── __init__.py
│       ├── spline.py       # Gaussian kernel utilities
│       └── layer.py        # WKNLayer (single network layer)
└── examples/
    └── demo.py             # Quick-start demo script
```

## Citation

If you use Wahkon in your research, please cite:

```bibtex
@article{wahkon2026,
  title   = {Wahkon: A Statistically Principled Deep RKHS Superposition Network},
  author  = {Chen, Yongkai and others},
  journal = {Proceedings of STAIX 2026},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
