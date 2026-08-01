# Wahkon: A Statistically Principled Deep RKHS Superposition Network

Code accompanying the STAIX 2026 paper
[*"Wahkon: A Statistically Principled Deep RKHS Superposition Network."*](https://openreview.net/pdf?id=pbHIU8XWtM)

This repository contains all scripts needed to reproduce the simulation
experiments (Figures and Tables) in the paper.


## Environment Setup

Three separate environments are recommended to avoid conflicting
dependencies between the KAN baseline (official
pykan), the NTK baseline (neural-tangents / JAX), and wahkon.

### 1. Wahkon environment (main)

Runs Wahkon, MLP, MLP (Deep), BNN, and all plotting scripts.
Python >= 3.10 is required.

```bash
conda create -n wahkon python=3.10 -y
conda activate wahkon

# PyTorch (adjust for your CUDA version — see https://pytorch.org)
pip install torch

# Other dependencies
pip install numpy pandas matplotlib scikit-learn scipy tqdm filelock
```

Verify the install:

```bash
python -c "from wahkon_backbone import WahkonBase; print('OK')"
```

### 2. KAN baseline environment

The official [pykan](https://github.com/KindXiaoming/pykan) package
(used for the KAN baseline in `run_kan_pykan.py`) requires a separate
environment because of JAX/neural-tangents incompatibilities:

```bash
conda create -n kan_baseline python=3.10 -y
conda activate kan_baseline
pip install torch numpy pandas filelock pykan scikit-learn scipy
```

Verify:

```bash
python -c "from kan import KAN; print('OK')"
```

### 3. NTK baseline environment

The NTK baseline (`run_ntk_nt_bo.py`) uses
[neural-tangents](https://github.com/google/neural-tangents) (JAX-based).

```bash
conda create -n ntk python=3.10 -y
conda activate ntk
pip install jax jaxlib neural-tangents
pip install torch numpy pandas filelock scikit-learn scipy
```

Verify:

```bash
python -c "import neural_tangents as nt; print('OK')"
```


## File Overview

| File | Description |
|---|---|
| `wahkon_backbone.py` | Self-contained Wahkon backbone: Gaussian kernel utilities, WahkonLayer, LBFGS optimizer, WahkonBase class, dataset helpers |
| `wahkon.py` | Core Wahkon implementation (`ProfileWKN`): profile objective + GP posterior |
| `compare_methods.py` | Method runners: `run_wahkon`, `run_mlp`, `run_mlp_deep`, `run_bnn` |
| `run_repeated.py` | RMSE + CI experiments for Wahkon, MLP, MLP (Deep), BNN |
| `run_kan_pykan.py` | KAN baseline experiments (standalone, uses official pykan) |
| `run_ntk_nt_bo.py` | NTK baseline experiments (standalone, uses neural-tangents) |
| `plot_results.py` | Plot log₁₀(RMSE) vs n_train comparison figure |
| `plot_ci_combined.py` | Plot CI width and coverage (Wahkon, NTK, BNN) |
| `prior_visual.py` | Monte Carlo Q-Q plots for Wahkon prior normality check |
| `requirement.txt` | Python package dependencies |


## Methods

| Method | Description |
|---|---|
| Wahkon | Deep RKHS superposition network with profile objective and GP posterior |
| MLP | Standard multilayer perceptron |
| MLP (Deep) | Higher-capacity MLP variant with 10× width and 2× depth |
| BNN | Bayes by Backprop (mean-field Gaussian variational posterior) |
| KAN | Kolmogorov-Arnold Network with B-spline basis (official pykan) |
| NTK | Neural Tangent Kernel as GP posterior baseline (neural-tangents / JAX) |


## Test Functions

Four test functions (f1–f4), each evaluated at
n_train ∈ {100, 200, 400, 800, 1600, 3200, 10000} with 100 random seeds.

| Name | Formula | D |
|---|---|---|
| f1 | log(x₁²+x₂²+\|tan(x₃)\|) + cot(π/(1+exp(x₁²+sin(6x₂)+x₃²))) | 3 |
| f2 | sin(x₁²+x₂²+…+x₁₀²) | 10 |
| f3 | exp(0.5·(sin(π(x₁²+x₂²)) + sin(π(x₃²+x₄²)))) | 4 |
| f4 | exp(sin(π(x₁²+x₂²)))·cos(π·x₃·x₄) | 6 |


## Reproducing Results

All commands below assume you are in this directory.
Set `OUT_DIR` to where results should be saved:

```bash
OUT_DIR=./results
```

### Step 1: Wahkon + MLP + MLP (Deep) + BNN

```bash
conda activate wahkon
python run_repeated.py --out_dir $OUT_DIR
```

Defaults: experiments f1–f4, training sizes 100–10000, 100 seeds, all CPUs.

### Step 2: KAN baseline

```bash
conda activate kan_baseline
python run_kan_pykan.py --out_dir $OUT_DIR
```

### Step 3: NTK baseline

```bash
conda activate ntk
python run_ntk_nt_bo.py --out_dir $OUT_DIR
```

### Step 4: Plot RMSE comparison (Figure in paper)

```bash
conda activate wahkon
python plot_results.py --results_dir $OUT_DIR --combined
```

Output: `combined_log_rmse.pdf` in `$OUT_DIR`.

### Step 5: Plot CI width and coverage

```bash
python plot_ci_combined.py --results_dir $OUT_DIR
```

Output: `combined_ci_grid.png` and individual metric plots in `$OUT_DIR`.

### Step 6: Prior visualization (Q-Q plot)

```bash
python prior_visual.py
```

Output: `figure/width_4_overlaid_qq_plots.png` and
`figure/width_4_prior_md_hist.png`.


## HPC Usage

For cluster runs, all three runner scripts support parallel execution
and crash-safe resume. If a run is interrupted, re-run the same command
and completed seeds are skipped automatically.

```bash
# Example: run on 100 cores
python run_repeated.py --out_dir $OUT_DIR --n_jobs 100
python run_kan_pykan.py --out_dir $OUT_DIR --n_jobs 100
python run_ntk_nt_bo.py --out_dir $OUT_DIR --n_jobs 100
```

To re-run only specific methods (overriding old results):

```bash
python run_repeated.py --methods bnn --override --out_dir $OUT_DIR
```


## Common Arguments

All three runner scripts share these arguments:

| Argument | Default | Description |
|---|---|---|
| `--expname` | `f1 f2 f3 f4` | Experiments to run |
| `--n_train` | `100 200 400 800 1600 3200 10000` | Training set sizes |
| `--n_seeds` | `100` | Number of random seeds |
| `--n_jobs` | `-1` (all cores) | Parallel workers |
| `--out_dir` | `./results` | Output directory |
| `--override` | off | Re-run and replace existing results |


## Output Format

Each runner produces CSV files named `{expname}_n{n_train}_log.csv`
with columns: `seed, method, rmse, y_width_at_95`.

All methods write to the same CSV format, so results from different
scripts (and environments) can be combined by pointing the plotting
scripts to the same `--out_dir` / `--results_dir`.
