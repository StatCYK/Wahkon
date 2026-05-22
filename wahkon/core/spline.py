"""
spline.py  --  Wahkon Gaussian kernel utilities
================================================
"""

import torch


# ---------------------------------------------------------------------------
# Gaussian (RBF / smoothing-spline) kernel
# ---------------------------------------------------------------------------

def SS_batch(x, grid, sigma=1.0, device='cpu'):
    """
    Evaluate the Gaussian (RBF / smoothing-spline) kernel on a grid.

    Used in the Wahkon RKHS forward pass: each edge function is represented as
        phi(t) = sum_i  c_i  K(grid_i, t),   K(s,t) = exp(-(s-t)^2 / sigma^2).

    Args
    ----
    x      : (size, batch)  -- spline-first layout
    grid   : (size, G+1)
    sigma  : float -- kernel bandwidth
    device : str

    Returns
    -------
    value  : (size, G+1, batch)  --  kernel evaluations
    """
    grid  = grid.unsqueeze(dim=2).to(device)   # (size, G+1, 1)
    x     = x.unsqueeze(dim=1).to(device)      # (size, 1, batch)
    value = torch.exp(-0.5 * (x - grid) ** 2 / sigma ** 2)
    return value


def SS_coef2curve(x_eval, grid, coef, sigma=1.0, device='cpu'):
    """
    Evaluate an RKHS function phi(t) = sum_i c_i * K(x_i, t)
    at new test points using the Gaussian kernel.

    Args
    ----
    x_eval : (size, batch_test)  -- evaluation points (spline-first)
    grid   : (size, n_train)     -- training knot positions
    coef   : (size, n_train)     -- representer coefficients c_i
    sigma  : float               -- kernel bandwidth
    device : str

    Returns
    -------
    y_eval : (size, batch_test)
    """
    K      = SS_batch(x_eval, grid, sigma=sigma, device=device)  # (size, n_train, batch_test)
    y_eval = torch.einsum('ij,ijk->ik', coef, K)
    return y_eval


def SS_curve2coef(x_eval, y_eval, grid, sigma, device='cpu', lamb=1e-8):
    """
    Convert function values to Gaussian-kernel representer coefficients
    via regularised least squares.

    Args
    ----
    x_eval : (batch, in_dim)   -- sample points
    y_eval : (batch, in_dim, out_dim)  -- function values
    grid   : (in_dim, num)     -- kernel centres
    sigma  : float             -- kernel bandwidth
    device : str
    lamb   : float             -- ridge regularisation (default 1e-8)

    Returns
    -------
    coef : (in_dim, out_dim, num)
    """
    batch   = x_eval.shape[0]
    in_dim  = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef  = grid.shape[1]

    # SS_batch with layout:
    # x_eval: (batch, in_dim) -> need (in_dim, batch) for SS_batch
    x_t = x_eval.permute(1, 0)                                    # (in_dim, batch)
    K   = SS_batch(x_t, grid, sigma, device=device)                # (in_dim, num, batch)
    mat = K.permute(0, 2, 1)                                       # (in_dim, batch, num)
    mat = mat[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)

    y_perm = y_eval.permute(1, 2, 0).unsqueeze(dim=3)             # (in_dim, out_dim, batch, 1)

    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), y_perm)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n, n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:, :, :, 0]

    return coef.to(device)
