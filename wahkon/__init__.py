"""
Wahkon -- A Statistically Principled Deep RKHS Superposition Network
====================================================================

Quick start::

    from wahkon import ProfileWKN, create_dataset

    import torch
    f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    dataset = create_dataset(f, n_var=2, train_num=500, test_num=200, seed=42)

    model = ProfileWKN(width=[2, 5, 1], grid=9, sigma=0.5)
    results, _, _ = model.train(dataset, steps=300, lr=0.005, lamb_last=0.01,
                                lamb_lower=0.01, batch=200, device='cpu')
"""

__version__ = "0.1.0"

from .profile import (
    WKN,
    ProfileWKN,
    create_dataset,
    create_dataset2,
    num_link_fun,
    lamb_scale,
    select_lambda_last_gcv,
    select_lambda_last_mll,
    select_lambda_profile_bo,
    select_lambda_profile_grid,
    select_lambda_twostage,
)

__all__ = [
    "ProfileWKN",
    "WKN",
    "create_dataset",
    "create_dataset2",
    "num_link_fun",
    "lamb_scale",
    "select_lambda_last_gcv",
    "select_lambda_last_mll",
    "select_lambda_profile_bo",
    "select_lambda_profile_grid",
    "select_lambda_twostage",
]
