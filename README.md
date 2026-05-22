# Wahkon

**A Statistically Principled Deep RKHS Superposition Network**

Wahkon unifies [Kolmogorov's superposition theorem](https://en.wikipedia.org/wiki/Kolmogorov%27s_theorem) with reproducing kernel Hilbert space (RKHS) regularization in the smoothing-spline tradition of Wahba. Each edge of the network carries a *learned univariate link function* constrained to lie in a common RKHS, yielding a framework that combines the representational power of deep architectures with the statistical rigor of kernel methods.

## Key Features

- **Deep Representer Theorem** — Despite infinite-dimensional function spaces, the estimator admits a finite kernel expansion at every layer.
- **Profile Objective** — Analytically concentrates last-layer coefficients for faster, stabler optimization.
- **Minimax-Optimal Rates** — Convergence guarantees via metric-entropy bounds under mild smoothness assumptions.

## Quick start: running wahkon on colab

demo notebook [demo_colab.ipynb](https://colab.research.google.com/drive/1L7jnFtuM1XiZL6-BJIiLFCbv6TP8voZr?usp=sharing) 

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

## Running the Demo

```bash
# Default: exp1 (f₁, D=3), 200 training samples
python examples/demo.py

# Custom settings
python examples/demo.py --expname exp3 --n_train 400 --seed 0

# Use GPU
python examples/demo.py --device cuda
```

The demo trains a ProfileWKN model, evaluates point prediction RMSE, and saves a summary plot to `./results/`.


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
    └── demo_colab.ipynb             # Quick-start colab-version demo script
```
