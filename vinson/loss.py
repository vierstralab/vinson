import torch
from torch.nn.functional import softplus


def negative_binomial_loss(input, target, r, scale_factor=1e6, reduction="mean"):
    """
    Compute the negative binomial loss from normalized density

    The 'input' and 'target' tensors are non-negative.  'input' is the
    prediction logit. 'target' is the normalized density
    (window tag count / sample read depth * scale factor). We convert these
    tensors to read counts by inversing the normalization transform.
    'eps' is a small value to avoid zeros.
    """

    # p =  (r - 1) / ( target_counts + r - 1) # sets the mode
    p = r / (r + input)  # sets mean
    phat = r / (r + target)

    # combiln = (
    #     torch.lgamma(target_counts + r)
    #     - torch.lgamma(target_counts + 1)
    #     - torch.lgamma(r)
    # )
    # nll = -(combiln + r * torch.log(p) + target_counts * torch.log1p(-p))
    
    nll = -(r * torch.log(p) + target * torch.log1p(-p))
    
    # nll = -(
    #     r * torch.log(p)
    #     + target * torch.log1p(-p)
    #     - r * torch.log(phat)
    #     - target * torch.log1p(-phat)
    # )

    if reduction == "mean":
        return nll.mean()
    else:
        return nll


def poisson_loss(input, target, reduction="mean"):
    """
    Compute the Poisson NLL from normalized density

    This loss function returns the relative NLL from the best
    possible one. This is because the Poisson NLL can be both
    postive and negative, which really messes up tracking
    training progress.
    """

    nll = -(target * (torch.log(input) - torch.log(target)) - input + target)

    if reduction == "mean":
        return nll.mean()
    else:
        return nll


def binomial_mixture_nll(input, target, n, d, tau):
    """
    Compute the binomial mixture loss considering the BAD score

    input: logit p
    target: ref counts
    n: total read depth
    d: log(bad_score)
    tau: 0.5*log(2)
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
    Compute the binomial mixture loss considering the BAD score

    This loss function returns the relative NLL from the best possible one.

    input: logit p
    target: ref counts
    n: total read depth
    bad: BAD score
    tau: 0.5*log(2)
    """
    d = torch.log(bad_score)
    nll = binomial_mixture_nll(input, target, n, d, tau)

    if reduction == "mean":
        return nll.mean()
    else:
        return nll
