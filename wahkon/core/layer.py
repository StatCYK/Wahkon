"""
layer.py  --  WKNLayer (Wahkon network layer)
==============================================
Each layer contains learned univariate link functions represented in
a Gaussian RKHS:  phi(t) = sum_i c_i K(grid_i, t).
"""

import torch
import torch.nn as nn
import numpy as np
from .spline import SS_batch, SS_curve2coef


def sparse_mask(in_dim, out_dim):
    """Return a (in_dim, out_dim) mask with ~sqrt(in_dim*out_dim) ones."""
    mask = torch.zeros(in_dim, out_dim)
    if in_dim == out_dim:
        mask = torch.eye(in_dim)
    else:
        n_ones = max(1, int(np.sqrt(in_dim * out_dim)))
        idx    = torch.randperm(in_dim * out_dim)[:n_ones]
        mask.view(-1)[idx] = 1.0
    return mask


class WKNLayer(nn.Module):
    """
    One layer of a Wahkon (WKN) network.

    Uses the Gaussian kernel basis  K(s,t) = exp(-(s-t)^2 / sigma^2).

    Attributes
    ----------
    grid       : (in_dim, num)        -- kernel centres
    coef       : (in_dim, out_dim, num)  -- representer coefficients
    scale_base : (in_dim, out_dim)    -- residual (base_fun) scale
    scale_sp   : (in_dim, out_dim)    -- kernel scale
    mask       : (in_dim, out_dim)    -- binary pruning mask
    base_fun   : callable
    sigma      : float -- kernel bandwidth
    """

    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        noise_scale=0.5,
        scale_base_mu=0.0,
        scale_base_sigma=1.0,
        scale_sp=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=(-1, 1),
        sp_trainable=True,
        sb_trainable=True,
        save_plot_data=True,
        device='cpu',
        sparse_init=False,
        sigma=1.0,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.num     = num
        self.save_plot_data = save_plot_data
        self.sigma   = sigma

        # ---- Gaussian kernel grid & coefficients ---------------------
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num)[None, :].expand(in_dim, num)
        self.grid = nn.Parameter(grid.clone(), requires_grad=False)

        # Representer coefficients initialised via SS_curve2coef
        noises = torch.randn(num, in_dim, out_dim) * noise_scale / num / in_dim
        noises = noises.to(device)
        self.coef = nn.Parameter(
            SS_curve2coef(self.grid.permute(1, 0), noises, self.grid,
                          sigma, device=device)
        )
        # coef shape: (in_dim, out_dim, num)

        # ---- pruning mask ------------------------------------------------
        if sparse_init:
            self.mask = nn.Parameter(sparse_mask(in_dim, out_dim), requires_grad=False)
        else:
            self.mask = nn.Parameter(torch.ones(in_dim, out_dim), requires_grad=False)

        # ---- trainable scales -------------------------------------------
        self.scale_base = nn.Parameter(
            scale_base_mu / np.sqrt(in_dim)
            + scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) / np.sqrt(in_dim),
            requires_grad=sb_trainable,
        )
        self.scale_sp = nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp,
            requires_grad=sp_trainable,
        )

        self.base_fun  = base_fun
        self.grid_eps  = grid_eps

        self.to(device)

    # ----------------------------------------------------------------------
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    # ----------------------------------------------------------------------
    def _gaussian_coef2curve(self, x):
        """Evaluate Gaussian kernel basis:  phi(x) = sum_i c_i K(grid_i, x).

        Parameters
        ----------
        x : (batch, in_dim)

        Returns
        -------
        y : (batch, in_dim, out_dim)
        """
        x_t = x.permute(1, 0)                                     # (in_dim, batch)
        K = SS_batch(x_t, self.grid, sigma=self.sigma,
                     device=x.device)                              # (in_dim, num, batch)
        y = torch.einsum('ioj,ijb->bio', self.coef, K)            # (batch, in_dim, out_dim)
        return y

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, in_dim)

        Returns
        -------
        y          : (batch, out_dim)
        preacts    : (batch, out_dim, in_dim)
        postacts   : (batch, out_dim, in_dim)
        postspline : (batch, out_dim, in_dim)
        """
        batch   = x.shape[0]
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x)                                    # (batch, in_dim)
        y = self._gaussian_coef2curve(x)                           # (batch, in_dim, out_dim)

        postspline = y.clone().permute(0, 2, 1)                    # (batch, out_dim, in_dim)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y                              # (batch, in_dim, out_dim)

        postacts = y.clone().permute(0, 2, 1)                      # (batch, out_dim, in_dim)
        y        = torch.sum(y, dim=1)                             # (batch, out_dim)

        return y, preacts, postacts, postspline

    # ----------------------------------------------------------------------
    def update_grid_from_samples(self, x, mode='sample'):
        """
        Adapt the grid to the distribution of incoming activations.

        Parameters
        ----------
        x    : (batch, in_dim)
        mode : 'sample' (default) or 'grid'
        """
        outlier_pt = 0.05
        num    = self.num
        device = x.device

        y_eval = self._gaussian_coef2curve(x)                      # (batch, in_dim, out_dim)
        x_eval = x.permute(1, 0)                                   # (in_dim, batch)

        grid_range = torch.quantile(
            x_eval,
            torch.tensor([outlier_pt / 2, 1 - outlier_pt / 2]),
            dim=1, keepdim=True,
        )
        anchors = torch.einsum(
            'i,j->ij',
            torch.ones(x_eval.shape[0], device=device),
            torch.linspace(0, 1, steps=num, device=device),
        )
        anchors = anchors * (grid_range[1, :, :] - grid_range[0, :, :]) + grid_range[0, :, :]
        anchors = anchors.unsqueeze(dim=1).to(device)
        x_eval  = x_eval.unsqueeze(dim=2).to(device)
        dist      = torch.abs(x_eval - anchors)
        grids_idx = torch.argmin(dist, dim=1)
        new_grid  = torch.cat(
            [x_eval[i, grids_idx[i, :], :] for i in range(x_eval.shape[0])], -1
        ).T                                                        # (in_dim, num)

        self.grid.data = new_grid
        self.coef.data = SS_curve2coef(
            x_eval[:, :, 0].permute(1, 0), y_eval, self.grid,
            self.sigma, device=device,
        )

    # ----------------------------------------------------------------------
    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        """Initialise grid from a coarser parent WKNLayer."""
        x_pos  = torch.sort(x, dim=0)[0]

        y_eval = parent._gaussian_coef2curve(x_pos)                # (batch, in_dim, out_dim)
        num    = self.num
        device = x.device

        outlier_pt = 0.05
        x_eval = x_pos.permute(1, 0)                              # (in_dim, batch)
        grid_range = torch.quantile(
            x_eval,
            torch.tensor([outlier_pt / 2, 1 - outlier_pt / 2]),
            dim=1, keepdim=True,
        )
        anchors = torch.einsum(
            'i,j->ij',
            torch.ones(x_eval.shape[0], device=device),
            torch.linspace(0, 1, steps=num, device=device),
        )
        anchors = anchors * (grid_range[1, :, :] - grid_range[0, :, :]) + grid_range[0, :, :]
        anchors = anchors.unsqueeze(dim=1).to(device)
        x_eval  = x_eval.unsqueeze(dim=2).to(device)
        dist      = torch.abs(x_eval - anchors)
        grids_idx = torch.argmin(dist, dim=1)
        new_grid  = torch.cat(
            [x_eval[i, grids_idx[i, :], :] for i in range(x_eval.shape[0])], -1
        ).T

        self.grid.data = new_grid
        self.coef.data = SS_curve2coef(
            x_eval[:, :, 0].permute(1, 0), y_eval, self.grid,
            self.sigma, device=device,
        )

    # ----------------------------------------------------------------------
    def get_subset(self, in_id, out_id):
        """Return a new WKNLayer restricted to the given neuron subsets."""
        layer = WKNLayer(
            in_dim=len(in_id), out_dim=len(out_id),
            num=self.num,
            scale_base_mu=0.0, scale_base_sigma=0.0,
            scale_sp=0.0,
            base_fun=self.base_fun,
            grid_eps=self.grid_eps,
            device=self.device,
            sigma=self.sigma,
        )
        layer.grid.data       = self.grid[in_id, :]
        layer.coef.data       = self.coef[in_id, :, :][:, out_id, :]
        layer.scale_base.data = self.scale_base[in_id, :][:, out_id]
        layer.scale_sp.data   = self.scale_sp[in_id, :][:, out_id]
        layer.mask.data       = self.mask[in_id, :][:, out_id]
        return layer
