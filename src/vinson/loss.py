import torch
from torch.nn.functional import softplus
import math


def mse_loss(input, target, reduction="mean"):
    """
    Compute the mean squared error (MSE) loss between the log-transformed input and target tensors.

    Parameters
    ----------
    input : torch.Tensor
        Predicted values (must be positive).
    target : torch.Tensor
        Ground truth values (must be positive).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' (default) returns a scalar tensor,
        'none' returns a tensor of losses per element.

    Returns
    -------
    torch.Tensor
        The computed MSE loss. If reduction is 'mean', returns a scalar tensor;
        otherwise, returns a tensor of losses per element.

    Notes
    -----
    - Applies a log transformation to both input and target before computing squared error.
    - Useful for comparing predicted and observed values on a multiplicative scale.
    """
    log_target = torch.log(target)
    log_input = torch.log(input)
    
    nll = torch.square(log_target - log_input)

    return nll.mean() if reduction == "mean" else nll


def negative_binomial_loss(
    input, target, r, relative=True, method="mean", reduction="mean"
):
    """
    Compute the negative binomial loss from raw read counts.

    Parameters
    ----------
    input : torch.Tensor
        Predicted counts.
    target : torch.Tensor
        Observed counts.
    r : float or torch.Tensor
        Dispersion parameter of the negative binomial distribution.
    method : str, optional
        Parameterization method for the negative binomial ('mean' or 'mode'). Default is 'mean'.
    relative : bool, optional
        If True, computes the relative negative log-likelihood (NLL) compared to the best possible prediction.
        If False, computes the standard NLL. Default is True.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' or 'none' (default is 'mean').

    Returns
    -------
    torch.Tensor
        The computed negative binomial loss. If reduction is 'mean', returns a scalar tensor;
        otherwise, returns a tensor of losses per element.

    Notes
    -----
    - Uses either mean or mode parameterization for the negative binomial distribution.
    - If relative is True, computes the difference in NLL between prediction and observed counts.
    - Useful for modeling overdispersed count data, such as sequencing reads.
    """
    if method == "mode":
        p = (r - 1) / (input + r - 1)
        phat = (r - 1) / (input + r - 1)
    else:
        p = r / (r + input)  # sets mean
        phat = r / (r + target)

    if relative:
        nll = -(
            r * torch.log(p)
            + target * torch.log1p(-p)
            - r * torch.log(phat)
            - target * torch.log1p(-phat)
        )
    else:
        nll = -(r * torch.log(p) + target * torch.log1p(-p))

    return nll.mean() if reduction == "mean" else nll


class PoissonNLL(torch.nn.Module):
    """Relative Poisson negative log-likelihood with clipped exponential"""
    def __init__(self, exp_clip=30.0, pseudocount=1e-6, log_input=True, reduction="mean", dtype=torch.float32):
        """
        Parameters
        ----------
        exp_clip : float, optional
            Clipping value for the exponential function to prevent overflow.
            Only used if log_input is True.
        pseudocount : float, optional
            Small constant added to input when log_input is False to prevent log(0).
            Only used if log_input is False.
        log_input : bool, optional
            If True, input is assumed to be in log space. If False, input is in normal space.
        reduction : str, optional
            Specifies the reduction to apply to the output: 'mean' or 'sum'
        dtype : torch.dtype, optional
            Data type for computations.
        """
        super(PoissonNLL, self).__init__()
        self.c = float(exp_clip)
        self.reduction = reduction
        self.dtype = dtype
        self.pseudocount = pseudocount
        self.log_input = log_input

    def clipped_exp(self, x: torch.Tensor) -> torch.Tensor:
        c_t = torch.as_tensor(self.c, dtype=x.dtype, device=x.device)
        rel = torch.relu(x - c_t)  # 0 if x<=c, x-c if x>c
        exp_bounded = torch.exp(x - rel)  # exp(x) or exp(c); never exp(large x)
        exp_c = torch.exp(c_t)
        return exp_bounded + exp_c * rel  # e^x (x<=c), e^c(1+x-c) (x>c)

    def forward(self, input, target):
        if self.log_input:
            log_input = input
            lam = self.clipped_exp(log_input)
        else:
            log_input = torch.log(input + self.pseudocount)
            lam = input

        # Poisson NLL centered at target
        nll = (lam - target) + torch.special.xlogy(target, target) - target * log_input

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        else:
            return nll


def binomial_mixture_nll(input, target, n, d, tau):
    """
    Compute the negative log-likelihood (NLL) for a binomial mixture model with BAD score adjustment.

    Parameters
    ----------
    input : torch.Tensor
        Logit of the predicted probability.
    target : torch.Tensor
        Reference counts (number of successes).
    n : torch.Tensor or float
        Total read depth (number of trials).
    log_bad_score : torch.Tensor or float
        Logarithm of the BAD score.
    tau : float
        Prior scale parameter (e.g., 0.5 * log(2)). If tau > 0, a quadratic prior is added.

    Returns
    -------
    torch.Tensor
        The computed binomial mixture negative log-likelihood. Returns a tensor of losses per element.

    Notes
    -----
    The loss is computed by combining two mixture components, each weighted by the BAD score.
    If tau > 0, a quadratic prior term is added to encourage regularization.
    """
    Lp = target * (input + d) - n * softplus(input + d)
    Lm = target * (input - d) - n * softplus(input - d)
    nll = -torch.logsumexp(torch.stack([Lp, Lm], dim=-1), dim=-1)

    if tau > 0:
        prior = input.pow(2) / (2 * tau**2)
        nll += prior

    return nll


def binomial_mixture_loss(input, target, n, bad_score, tau=-1, reduction="mean"):
    """
    Compute the binomial mixture loss with BAD score adjustment.

    Parameters
    ----------
    input : torch.Tensor
        Logit of the predicted probability.
    target : torch.Tensor
        Reference counts (number of successes).
    n : torch.Tensor or float
        Total read depth (number of trials).
    bad_score : torch.Tensor or float
        BAD score for each observation.
    tau : float, optional
        Prior scale parameter (e.g., 0.5 * log(2)). If tau > 0, a quadratic prior is added. Default is -1 (no prior).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' or 'none' (default is 'mean').

    Returns
    -------
    torch.Tensor
        The computed binomial mixture loss. If reduction is 'mean', returns a scalar tensor;
        otherwise, returns a tensor of losses per element.

    Notes
    -----
    The loss is computed as the relative NLL from the best possible prediction,
    combining two mixture components weighted by the BAD score. If tau > 0, a quadratic prior term is added.
    """
    d = torch.log(bad_score)
    nll = binomial_mixture_nll(input, target, n, d, tau)

    if reduction == "mean":
        return nll.mean()
    else:
        return nll


def binomial_mixture_nll_grad_hess(logit, k, n, d, tau):
    """
    Compute the negative log-likelihood, gradient, and Hessian for the binomial mixture model.

    Parameters
    ----------
    logit : torch.Tensor
        Logit of the predicted probability.
    k : torch.Tensor
        Reference counts (number of successes).
    n : torch.Tensor or float
        Total read depth (number of trials).
    d : torch.Tensor or float
        Logarithm of the BAD score.
    tau : float, optional
        Prior scale parameter. If tau > 0, a quadratic prior is added.

    Returns
    -------
    nll : torch.Tensor
        Negative log-likelihood.
    grad : torch.Tensor
        Gradient of the NLL with respect to logit.
    hess : torch.Tensor
        Hessian of the NLL with respect to logit.
    """
    logits = torch.stack([logit - d, logit + d], dim=0)
    probs = torch.sigmoid(logits)

    grad_components = k - n * probs
    hess_components = -n * probs * (1 - probs)

    ll = k * logits - n * softplus(logits)
    ll_sum = torch.logsumexp(ll, dim=0)

    weights = torch.exp(ll - ll_sum)
    variance = (weights * grad_components**2).sum(dim=0) - (
        (weights * grad_components).sum(dim=0)
    ) ** 2

    nll = -ll_sum
    grad = -(weights * grad_components).sum(dim=0)
    hess = -(variance + (weights * hess_components).sum(dim=0))

    if tau > 0:
        nll = nll + logit.pow(2) / (2 * tau**2)
        grad = grad + logit / (tau**2)
        hess = hess + 1 / (tau**2)

    return nll, grad, hess


def binomial_mixture_normed_loss(
    input, target, n, bad_score, tau=0, iters=5, max_step_frac=0.5, reduction="mean"
):
    """
    Compute the normalized negative log-likelihood (NLL) for the binomial mixture model.

    Parameters
    ----------
    input : torch.Tensor
        Logit of the predicted probability.
    target : torch.Tensor
        Reference counts (number of successes).
    n : torch.Tensor or float
        Total read depth (number of trials).
    bad_score : torch.Tensor or float
        BAD score for each observation.
    tau : float, optional
        Prior scale parameter. If tau > 0, a quadratic prior is added. Default is 0.
    iters : int, optional
        Number of Newton-Raphson iterations for root finding (default is 5).
    max_step_frac : float, optional
        Fraction of the mean distance from each root to the initial guesses that any
        single NR step can move. Prevents large jumps. Default: 0.5.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' or 'none' (default is 'mean').

    Returns
    -------
    torch.Tensor
        The normalized negative log-likelihood (NLL) for the input logits. If reduction is 'mean', returns a scalar tensor;
        otherwise, returns a tensor of losses per element.

    Notes
    -----
    - Finds the minimum NLL by optimizing the logit using Newton-Raphson iterations.
    - Returns the difference between the NLL at the input and the minimum NLL for normalization.
    - Useful for comparing model predictions to the best possible prediction for each observation.
    """

    d = torch.log(bad_score)

    # Initial roots: logit of empirical probability ± log_bad_score
    logit_p = torch.logit(target / n, eps=1e-6)
    initial_roots = torch.stack(
        [logit_p - 2 * d, logit_p - d, logit_p, logit_p + d, logit_p + 2 * d], dim=0
    )
    roots = initial_roots

    # Newton-Raphson optimization to refine roots
    for _ in range(iters):
        _, grad, hess = binomial_mixture_nll_grad_hess(roots, target, n, d, tau)
        max_step = (roots.unsqueeze(0) - initial_roots.unsqueeze(1)).abs().mean(
            dim=0
        ) * max_step_frac
        step = -grad / hess
        roots += torch.minimum(torch.abs(step), max_step) * torch.sign(step)

    # Compute NLL at roots and input
    nll_at_roots, _, _ = binomial_mixture_nll_grad_hess(roots, target, n, d, tau)
    min_nll = torch.min(nll_at_roots, dim=0)[0]

    nll_at_input, _, _ = binomial_mixture_nll_grad_hess(input, target, n, d, tau)

    #optional min nll fix
    if tau <= 0:
        edge = (target == 0) | (target == n)
        if edge.any():
            min_nll_edge = -math.log(2.0)
            min_nll = torch.where(edge, min_nll.new_tensor(min_nll_edge), min_nll)

    normalized_nll = nll_at_input - min_nll

    return normalized_nll.mean() if reduction == "mean" else normalized_nll
