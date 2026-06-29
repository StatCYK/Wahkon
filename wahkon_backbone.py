"""
wahkon_backbone.py -- Self-contained backbone for the Wahkon implementation
===========================================================================
Gaussian-kernel WahkonLayer, LBFGS optimizer, and WahkonBase class.
ProfileWKN (in wahkon.py) extends WahkonBase with the profile objective.

Contents:
    1. Gaussian kernel utilities  (SS_batch, SS_curve2coef)
    2. WahkonLayer                   (Gaussian kernel basis)
    3. LBFGS optimizer
    4. WahkonBase class
    5. Dataset helper              (create_dataset)
"""

import torch
import torch.nn as nn
import numpy as np
from functools import reduce


# ===================================================================
#  1. Gaussian kernel utilities
# ===================================================================

def SS_batch(x, grid, sigma=1.0, device='cpu'):
    """
    Evaluate the Gaussian (RBF / smoothing-spline) kernel on a grid.

    Used in the Wahkon RKHS forward pass: each edge function is represented as
        phi(t) = sum_i  c_i  K(grid_i, t),   K(s,t) = exp(-(s-t)^2 / sigma^2).

    Args
    ----
    x      : (size, batch)  -- spline-first layout (legacy, matches WahkonLayer v1 internals)
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

    # SS_batch with sskan2-compatible layout:
    # x_eval: (batch, in_dim) -> need (in_dim, batch) for our SS_batch
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


# ===================================================================
#  2. WahkonLayer
# ===================================================================

class WahkonLayer(nn.Module):
    """
    One layer of a Wahkon network (Gaussian kernel basis).

    Attributes
    ----------
    grid     : (in_dim, num)       -- kernel centres
    coef     : (in_dim, out_dim, num) -- representer coefficients
    scale_base : (in_dim, out_dim) -- residual (base_fun) scale
    scale_sp   : (in_dim, out_dim) -- kernel scale
    mask       : (in_dim, out_dim) -- binary pruning mask
    base_fun   : callable
    sigma      : float             -- kernel bandwidth
    device     : str
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
        device='cpu',
        sigma=1.0,
    ):
        """
        Parameters
        ----------
        scale_base_mu    : float -- mean of N(mu, sigma^2) for scale_base init
        scale_base_sigma : float -- std  of N(mu, sigma^2) for scale_base init
                           Set sigma=0 to disable the base-function branch.
        scale_sp         : float -- initial scale for the kernel branch.
        sigma            : float -- Gaussian kernel bandwidth
        """
        super().__init__()

        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.num     = num
        self.sigma   = sigma

        # ---- Gaussian kernel grid & coefficients ---------------------
        # Grid: num equally spaced kernel centres (no extension needed)
        grid = torch.linspace(-1, 1, steps=num)[None, :].expand(in_dim, num)
        self.grid = nn.Parameter(grid.clone(), requires_grad=False)
        # grid shape: (in_dim, num)

        # Representer coefficients initialised via SS_curve2coef
        noises = torch.randn(num, in_dim, out_dim) * noise_scale / num / in_dim
        noises = noises.to(device)
        self.coef = nn.Parameter(
            SS_curve2coef(self.grid.permute(1, 0), noises, self.grid,
                          sigma, device=device)
        )
        # coef shape: (in_dim, out_dim, num)

        # ---- pruning mask ------------------------------------------------
        self.mask = nn.Parameter(torch.ones(in_dim, out_dim), requires_grad=False)

        # ---- trainable scales -------------------------------------------
        self.scale_base = nn.Parameter(
            scale_base_mu / np.sqrt(in_dim)
            + scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) / np.sqrt(in_dim),
            requires_grad=True,
        )
        # sskan2 original: no 1/sqrt(in_dim) scaling on scale_sp
        self.scale_sp = nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp,
            requires_grad=True,
        )

        self.base_fun  = base_fun

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
        # x: (batch, in_dim) -> transpose to (in_dim, batch) for SS_batch
        x_t = x.permute(1, 0)                                     # (in_dim, batch)
        # SS_batch expects (size, batch) and grid (size, G+1)
        K = SS_batch(x_t, self.grid, sigma=self.sigma,
                     device=x.device)                              # (in_dim, G+1, batch)
        # coef: (in_dim, out_dim, G+1)
        # K:    (in_dim, G+1, batch)
        # want: (batch, in_dim, out_dim)
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

        y = self._gaussian_coef2curve(x)
        # y: (batch, in_dim, out_dim)

        postspline = y.clone().permute(0, 2, 1)                    # (batch, out_dim, in_dim)

        # scale_base : (in_dim, out_dim);  base : (batch, in_dim)
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
        # Matching sskan2 SS_update_grid_from_samples
        outlier_pt = 0.05
        num    = self.num
        device = x.device

        y_eval = self._gaussian_coef2curve(x)                  # (batch, in_dim, out_dim)
        x_eval = x.permute(1, 0)                               # (in_dim, batch)

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
        ).T                                                    # (in_dim, num)

        self.grid.data = new_grid
        self.coef.data = SS_curve2coef(
            x_eval[:, :, 0].permute(1, 0), y_eval, self.grid,
            self.sigma, device=device,
        )

    # ----------------------------------------------------------------------
    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        """Initialise grid from a coarser parent WahkonLayer."""
        x_pos  = torch.sort(x, dim=0)[0]

        y_eval = parent._gaussian_coef2curve(x_pos)            # (batch, in_dim, out_dim)
        num    = self.num
        device = x.device

        # Use same percentile-based grid selection as update_grid
        outlier_pt = 0.05
        x_eval = x_pos.permute(1, 0)                          # (in_dim, batch)
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
        """Return a new WahkonLayer restricted to the given neuron subsets."""
        layer = WahkonLayer(
            in_dim=len(in_id), out_dim=len(out_id),
            num=self.num,
            scale_base_mu=0.0, scale_base_sigma=0.0,
            scale_sp=0.0,
            base_fun=self.base_fun,
            device=self.device,
            sigma=self.sigma,
        )
        layer.grid.data       = self.grid[in_id, :]
        layer.coef.data       = self.coef[in_id, :, :][:, out_id, :]
        layer.scale_base.data = self.scale_base[in_id, :][:, out_id]
        layer.scale_sp.data   = self.scale_sp[in_id, :][:, out_id]
        layer.mask.data       = self.mask[in_id, :][:, out_id]
        return layer


# ===================================================================
#  3. LBFGS optimizer
# ===================================================================

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old low becomes new high
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    if len(bracket) == 1:
        t = bracket[0]
        f_new = bracket_f[0]
        g_new = bracket_g[0]
    else:
        t = bracket[low_pos]
        f_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class LBFGS(torch.optim.Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            tolerance_ys=tolerance_ys,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        device = views[0].device
        return torch.cat(views, dim=0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """

        torch.manual_seed(0)

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        tolerance_ys = group['tolerance_ys']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > tolerance_ys:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss


# ===================================================================
#  4. WahkonBase class
# ===================================================================

class WahkonBase(nn.Module):
    """
    Lightweight Wahkon base class (Gaussian kernel basis only).
    ProfileWKN inherits from this and adds the profile objective.
    """

    def __init__(
        self,
        width=None,
        grid=3,
        sigma=1.0,
        noise_scale=0.1,
        noise_scale_base=1.0,
        base_fun=torch.nn.SiLU(),
        scale_base_mu=0.0,
        scale_base_sigma=None,
        bias_trainable=True,
        grid_range=(-1, 1),
        sp_trainable=True,
        sb_trainable=True,
        device='cpu',
        seed=0,
        norm_type=None,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        if width is None:
            width = [2, 5, 1]

        self.width   = list(width)
        self.depth   = len(width) - 1
        self.grid    = grid
        self.sigma   = sigma
        self.base_fun = base_fun
        self.device  = device

        # Use noise_scale_base as the default for scale_base_sigma
        if scale_base_sigma is None:
            scale_base_sigma = noise_scale_base

        self.act_fun    = nn.ModuleList()
        self.biases     = nn.ModuleList()

        for l in range(self.depth):
            layer = WahkonLayer(
                in_dim=width[l], out_dim=width[l + 1],
                num=grid,
                noise_scale=noise_scale,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=1.0,
                base_fun=base_fun,
                device=device,
                sigma=sigma,
            )
            self.act_fun.append(layer)

            bias = nn.Linear(width[l + 1], 1, bias=False, device=device)
            bias.weight.data *= 0.0
            bias.requires_grad_(bias_trainable)
            self.biases.append(bias)

        # -- BatchNorm layers -----------------------------------------------
        # Applied between hidden layers (not after the final output layer).
        self.norm_type   = norm_type
        self.norm_layers = nn.ModuleList()
        if norm_type == 'batch':
            for l in range(self.depth - 1):   # no norm after the last layer
                d_out = width[l + 1]
                self.norm_layers.append(nn.BatchNorm1d(d_out, device=device))
        elif norm_type is not None:
            raise ValueError(f"norm_type must be None or 'batch'; got '{norm_type}'")

    # ------------------------------------------------------------------
    def forward(self, x):
        self.acts              = [x]
        self.spline_preacts    = []
        self.spline_postsplines = []
        self.spline_postacts   = []
        self.acts_scale        = []
        self.acts_scale_std    = []

        for l in range(self.depth):
            x_num, preacts, postacts, postspline = self.act_fun[l](x)

            x = x_num

            # input_range: broadcast (out_dim, in_dim) / (1, in_dim)
            input_range  = (self.act_fun[l].grid[:, -1] - self.act_fun[l].grid[:, 0] + 1e-4)  # (in_dim,)
            output_range = torch.mean(torch.abs(postacts), dim=0)                              # (out_dim, in_dim)
            self.acts_scale.append(output_range / input_range[None, :])
            self.acts_scale_std.append(torch.std(postacts, dim=0))

            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight

            # Apply normalisation between hidden layers only.
            if self.norm_layers and l < self.depth - 1:
                x = self.norm_layers[l](x)

            self.acts.append(x)

        return x

    # ------------------------------------------------------------------
    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])


# ===================================================================
#  5. Dataset helpers
# ===================================================================

def create_dataset(f, f_true=None, n_var=2, ranges=(-1, 1),
                   train_num=1000, test_num=1000,
                   normalize_input=False, normalize_label=False,
                   device='cpu', seed=0):
    """Create a train/test dictionary for a synthetic function ``f``."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(ranges, (list, tuple)) and not isinstance(ranges[0], (list, tuple)):
        ranges = [ranges] * n_var

    train_input = torch.zeros(train_num, n_var)
    test_input  = torch.zeros(test_num,  n_var)
    for i, r in enumerate(ranges):
        train_input[:, i] = torch.FloatTensor(train_num).uniform_(r[0], r[1])
        test_input[:, i]  = torch.FloatTensor(test_num).uniform_(r[0], r[1])

    train_label = f(train_input)
    test_label  = f(test_input)
    train_true  = f_true(train_input) if f_true is not None else train_label
    test_true   = f_true(test_input)  if f_true is not None else test_label

    if normalize_input:
        mean = train_input.mean(dim=0, keepdim=True)
        std  = train_input.std(dim=0, keepdim=True).clamp(min=1e-6)
        train_input = (train_input - mean) / std
        test_input  = (test_input  - mean) / std

    if normalize_label:
        mean = train_label.mean()
        std  = train_label.std().clamp(min=1e-6)
        train_label = (train_label - mean) / std
        test_label  = (test_label  - mean) / std
        train_true  = (train_true  - mean) / std
        test_true   = (test_true   - mean) / std

    return {
        'train_input': train_input.to(device),
        'train_label': train_label.to(device),
        'test_input':  test_input.to(device),
        'test_label':  test_label.to(device),
        'train_true':  train_true.to(device),
        'test_true':   test_true.to(device),
    }


