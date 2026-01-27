import torch
from torch.nn.functional import softplus
from typing import Union


class MSELoss(torch.nn.Module):
    """
    Compute the mean squared error (MSE) loss between the log-transformed input and target tensors.

    This loss applies a logarithmic transformation to both predicted and target values
    before computing the squared error, making it suitable for multiplicative comparisons.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' (default) returns a scalar tensor,
        'sum' returns the sum of losses, 'none' returns a tensor of losses per element.

    Notes
    -----
    - Applies a log transformation to both input and target before computing squared error.
    - Useful for comparing predicted and observed values on a multiplicative scale.
    """

    def __init__(self, reduction: str = "mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss on log-transformed values.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values (must be positive).
        target : torch.Tensor
            Ground truth values (must be positive).

        Returns
        -------
        torch.Tensor
            The computed MSE loss. If reduction is 'sum' or 'mean', returns a scalar tensor;
            otherwise, returns a tensor of losses per element.
        """
        log_target = torch.log(target)
        log_input = torch.log(input)

        nll = torch.square(log_target - log_input)

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        else:
            return nll


class NegativeBinomialLoss(torch.nn.Module):
    """
    Negative log-likelihood loss for the negative binomial distribution.

    This loss computes the NLL for observed counts assuming a negative binomial distribution,
    with options for relative loss and different parameterizations.

    Parameters
    ----------
    relative : bool, optional
        If True, computes the relative NLL compared to the best possible prediction (default is True).
    mom : str, optional
        Method of moments parameterization: 'mean' or 'mode' (default is 'mean').
    reduction : str, optional
        Specifies the reduction: 'mean', 'sum', or 'none' (default is 'mean').
    """

    def __init__(
        self, relative: bool = True, mom: str = "mean", reduction: str = "mean"
    ):
        super(NegativeBinomialLoss, self).__init__()
        self.reduction = reduction
        self.mom = mom
        self.relative = relative

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, r: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the negative binomial loss.

        Parameters
        ----------
        input : torch.Tensor
            Predicted counts.
        target : torch.Tensor
            Observed counts.
        r : float or torch.Tensor
            Dispersion parameter of the negative binomial distribution.

        Returns
        -------
        torch.Tensor
            The computed negative binomial loss. If reduction is 'sum' or 'mean', returns a scalar tensor;
            otherwise, returns a tensor of losses per element.
        """
        if self.method == "mode":
            p = (r - 1) / (input + r - 1)
            phat = (r - 1) / (input + r - 1)
        else:
            p = r / (r + input)
            phat = r / (r + target)

        if self.relative:
            nll = -(
                r * torch.log(p)
                + target * torch.log1p(-p)
                - r * torch.log(phat)
                - target * torch.log1p(-phat)
            )
        else:
            nll = -(r * torch.log(p) + target * torch.log1p(-p))

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        else:
            return nll


class PoissonNLLLoss(torch.nn.Module):
    """
    Negative log-likelihood loss for the Poisson distribution.

    This loss computes the relative Poisson NLL for observed counts, with options for
    input scaling and numerical stability via clipped exponential.

    Parameters
    ----------
    exp_clip : float, optional
        Clipping value for exponential to prevent overflow (default is 30.0).
    pseudocount : float, optional
        Small value added to input for log computation (default is 1e-6).
    log_input : bool, optional
        If True, input is treated as log(lambda); otherwise, as lambda (default is True).
    relative : bool, optional
        If True, computes relative NLL centered at target (default is True).
    reduction : str, optional
        Specifies the reduction: 'mean', 'sum', or 'none' (default is 'mean').
    """

    def __init__(
        self,
        exp_clip: float = 30.0,
        pseudocount: float = 1e-6,
        log_input: bool = True,
        relative: bool = True,
        reduction: str = "mean",
    ):
        super(PoissonNLLLoss, self).__init__()
        self.c = float(exp_clip)
        self.reduction = reduction
        self.pseudocount = pseudocount
        self.log_input = log_input
        self.relative = relative

    def clipped_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a numerically stable exponential function with clipping.

        For values below the clip threshold, computes exp(x). For values above,
        computes exp(c) * (1 + x - c) to prevent overflow.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Clipped exponential of x.
        """
        c_t = torch.as_tensor(self.c, dtype=x.dtype, device=x.device)
        rel = torch.relu(x - c_t)  # 0 if x<=c, x-c if x>c
        exp_bounded = torch.exp(x - rel)  # exp(x) or exp(c); never exp(large x)
        exp_c = torch.exp(c_t)
        return exp_bounded + exp_c * rel  # e^x (x<=c), e^c(1+x-c) (x>c)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Poisson negative log-likelihood loss.

        This method calculates the relative Poisson NLL, centered at the target values.
        It uses a clipped exponential for numerical stability when log_input is True.

        Parameters
        ----------
        input : torch.Tensor
            Predicted counts. If log_input is True, this is log(lambda); otherwise, lambda.
        target : torch.Tensor
            Observed counts.

        Returns
        -------
        torch.Tensor
            The computed Poisson loss. If reduction is 'sum' or 'mean', returns a scalar tensor;
            otherwise, returns a tensor of losses per element.
        """
        if self.log_input:
            log_input = input
            lam = self.clipped_exp(log_input)
        else:
            log_input = torch.log(input + self.pseudocount)
            lam = input

        # Poisson NLL centered at target
        if self.relative:
            nll = (
                (lam - target)
                + torch.special.xlogy(target, target)
                - target * log_input
            )
        else:
            raise NotImplementedError

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        else:
            return nll


PoissonNLL = PoissonNLLLoss # alias

class BinomialMixtureNLLLoss(torch.nn.Module):
    """
    Negative log-likelihood loss for a binomial mixture model with BAD score adjustment.

    This loss function models the data as a mixture of two binomial distributions,
    adjusted by a BAD score parameter, and optionally computes a relative loss
    normalized by the minimum achievable NLL.

    Parameters
    ----------
    relative : bool, optional
        If True, computes the relative NLL by subtracting the minimum NLL (default is True).
    tau : float, optional
        Regularization parameter (default is 0).
    iters : int, optional
        Number of Newton-Raphson iterations for root finding (default is 5).
    max_step_frac : float, optional
        Maximum step size fraction for root finding (default is 0.5).
    reduction : str, optional
        Specifies the reduction: 'mean', 'sum', or 'none' (default is 'mean').
    """

    def __init__(
        self,
        relative: bool = True,
        tau: float = 0,
        iters: int = 5,
        max_step_frac: float = 0.5,
        reduction: str = "mean",
    ):
        super(BinomialMixtureNLLLoss, self).__init__()
        self.relative = relative
        self.iters = iters
        self.max_step_frac = max_step_frac
        self.tau = tau
        self.reduction = reduction

    @classmethod
    def _nll_grad_hess(
        cls,
        input: torch.Tensor,
        target: torch.Tensor,
        n: torch.Tensor,
        log_bad: torch.Tensor,
        tau: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the negative log-likelihood (NLL), gradient, and Hessian for a binomial mixture model.

        This method evaluates the NLL and its derivatives for a mixture of two binomial distributions
        adjusted by a BAD score, with optional regularization.

        Parameters
        ----------
        input : torch.Tensor
            Logit of the predicted probability.
        target : torch.Tensor
            Observed counts.
        n : torch.Tensor
            Total number of counts.
        log_bad : torch.Tensor
            Logarithm of the BAD score.
        tau : float, optional
            Regularization parameter (default is 0).

        Returns
        -------
        tuple of torch.Tensor
            - nll : Negative log-likelihood.
            - grad : Gradient of the NLL with respect to input.
            - hess : Hessian of the NLL with respect to input.
        """
        logits = torch.stack([input - log_bad, input + log_bad], dim=0)
        probs = torch.sigmoid(logits)

        grad_components = target - n * probs
        hess_components = -n * probs * (1 - probs)

        ll = target * logits - n * softplus(logits)
        ll_sum = torch.logsumexp(ll, dim=0)

        weights = torch.exp(ll - ll_sum)
        variance = (weights * grad_components**2).sum(dim=0) - (
            (weights * grad_components).sum(dim=0)
        ) ** 2

        nll = -ll_sum
        grad = -(weights * grad_components).sum(dim=0)
        hess = -(variance + (weights * hess_components).sum(dim=0))

        if tau > 0:
            nll = nll + input.pow(2) / (2 * tau**2)
            grad = grad + input / (tau**2)
            hess = hess + 1 / (tau**2)

        return nll, grad, hess

    @classmethod
    def _find_roots(
        cls,
        target: torch.Tensor,
        n: torch.Tensor,
        log_bad: torch.Tensor,
        tau: float = 0,
        iters: int = 5,
        max_step_frac: float = 0.5,
    ) -> torch.Tensor:
        """
        Find roots of the negative log-likelihood function using Newton-Raphson optimization.

        This method initializes candidate roots based on the logit of the empirical probability
        and refines them through iterative Newton-Raphson steps to locate minima of the NLL.

        Parameters
        ----------
        target : torch.Tensor
            Observed counts.
        n : torch.Tensor
            Total number counts.
        log_bad : torch.Tensor
            Logarithm of the BAD score adjustment.
        tau : float, optional
            Regularization parameter (default is 0).
        iters : int, optional
            Number of Newton-Raphson iterations (default is 5).
        max_step_frac : float, optional
            Maximum step size as a fraction of the distance to initial roots (default is 0.5).

        Returns
        -------
        torch.Tensor
            Refined roots of the NLL function.
        """
        # Initial roots: logit of empirical probability ± log_bad_score
        logit_p = torch.logit(target / n)
        initial_roots = torch.stack(
            [
                logit_p - 2 * log_bad,
                logit_p - log_bad,
                logit_p,
                logit_p + log_bad,
                logit_p + 2 * log_bad,
            ],
            dim=0,
        )
        roots = initial_roots

        # Newton-Raphson optimization to refine roots
        for _ in range(iters):
            _, grad, hess = cls._nll_grad_hess(roots, target, n, log_bad, tau)
            max_step = (roots.unsqueeze(0) - initial_roots.unsqueeze(1)).abs().mean(
                dim=0
            ) * max_step_frac
            step = -grad / hess
            roots += torch.minimum(torch.abs(step), max_step) * torch.sign(step)

        return roots

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        n: torch.Tensor,
        bad_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the binomial mixture negative log-likelihood loss.

        This method calculates the NLL for a mixture of binomial distributions adjusted by a bad score.
        If relative is True, it normalizes the loss by subtracting the minimum NLL found via root finding.

        Parameters
        ----------
        input : torch.Tensor
            Logit of the predicted probability.
        target : torch.Tensor
            Observed counts.
        n : torch.Tensor
            Total number of counts.
        bad_score : torch.Tensor
            BAD score adjustment (will be log-transformed internally).

        Returns
        -------
        torch.Tensor
            The computed loss. If reduction is 'sum' or 'mean', returns a scalar tensor;
            otherwise, returns a tensor of losses per element.
        """
        log_bad = torch.log(bad_score)

        nll, _, _ = self._nll_grad_hess(input, target, n, log_bad, self.tau)

        if self.relative:
            # Find minimum NLL
            roots = self._find_roots(
                target, n, log_bad, self.tau, self.iters, self.max_step_frac
            )
            nll_at_roots, _, _ = self._nll_grad_hess(
                roots, target, n, log_bad, self.tau
            )
            min_nll = torch.min(nll_at_roots, dim=0)[0]
            # Substract min NLL to normalize
            nll -= min_nll

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        else:
            return nll


# def binomial_mixture_nll(input, target, n, d, tau):
#     """
#     Compute the negative log-likelihood (NLL) for a binomial mixture model with BAD score adjustment.

#     Parameters
#     ----------
#     input : torch.Tensor
#         Logit of the predicted probability.
#     target : torch.Tensor
#         Reference counts (number of successes).
#     n : torch.Tensor or float
#         Total read depth (number of trials).
#     log_bad_score : torch.Tensor or float
#         Logarithm of the BAD score.
#     tau : float
#         Prior scale parameter (e.g., 0.5 * log(2)). If tau > 0, a quadratic prior is added.

#     Returns
#     -------
#     torch.Tensor
#         The computed binomial mixture negative log-likelihood. Returns a tensor of losses per element.

#     Notes
#     -----
#     The loss is computed by combining two mixture components, each weighted by the BAD score.
#     If tau > 0, a quadratic prior term is added to encourage regularization.
#     """
#     Lp = target * (input + d) - n * softplus(input + d)
#     Lm = target * (input - d) - n * softplus(input - d)
#     nll = -torch.logsumexp(torch.stack([Lp, Lm], dim=-1), dim=-1)

#     if tau > 0:
#         prior = input.pow(2) / (2 * tau**2)
#         nll += prior

#     return nll

# def binomial_mixture_loss(input, target, n, bad_score, tau=-1, reduction="mean"):
#     """
#     Compute the binomial mixture loss with BAD score adjustment.

#     Parameters
#     ----------
#     input : torch.Tensor
#         Logit of the predicted probability.
#     target : torch.Tensor
#         Reference counts (number of successes).
#     n : torch.Tensor or float
#         Total read depth (number of trials).
#     bad_score : torch.Tensor or float
#         BAD score for each observation.
#     tau : float, optional
#         Prior scale parameter (e.g., 0.5 * log(2)). If tau > 0, a quadratic prior is added. Default is -1 (no prior).
#     reduction : str, optional
#         Specifies the reduction to apply to the output: 'mean' or 'none' (default is 'mean').

#     Returns
#     -------
#     torch.Tensor
#         The computed binomial mixture loss. If reduction is 'mean', returns a scalar tensor;
#         otherwise, returns a tensor of losses per element.

#     Notes
#     -----
#     The loss is computed as the relative NLL from the best possible prediction,
#     combining two mixture components weighted by the BAD score. If tau > 0, a quadratic prior term is added.
#     """
#     d = torch.log(bad_score)
#     nll = binomial_mixture_nll(input, target, n, d, tau)

#     if reduction == "mean":
#         return nll.mean()
#     else:
#         return nll


# def binomial_mixture_nll_grad_hess(logit, k, n, d, tau):
#     """
#     Compute the negative log-likelihood, gradient, and Hessian for the binomial mixture model.

#     Parameters
#     ----------
#     logit : torch.Tensor
#         Logit of the predicted probability.
#     k : torch.Tensor
#         Reference counts (number of successes).
#     n : torch.Tensor or float
#         Total read depth (number of trials).
#     d : torch.Tensor or float
#         Logarithm of the BAD score.
#     tau : float, optional
#         Prior scale parameter. If tau > 0, a quadratic prior is added.

#     Returns
#     -------
#     nll : torch.Tensor
#         Negative log-likelihood.
#     grad : torch.Tensor
#         Gradient of the NLL with respect to logit.
#     hess : torch.Tensor
#         Hessian of the NLL with respect to logit.
#     """
#     logits = torch.stack([logit - d, logit + d], dim=0)
#     probs = torch.sigmoid(logits)

#     grad_components = k - n * probs
#     hess_components = -n * probs * (1 - probs)

#     ll = k * logits - n * softplus(logits)
#     ll_sum = torch.logsumexp(ll, dim=0)

#     weights = torch.exp(ll - ll_sum)
#     variance = (weights * grad_components**2).sum(dim=0) - (
#         (weights * grad_components).sum(dim=0)
#     ) ** 2

#     nll = -ll_sum
#     grad = -(weights * grad_components).sum(dim=0)
#     hess = -(variance + (weights * hess_components).sum(dim=0))

#     if tau > 0:
#         nll = nll + logit.pow(2) / (2 * tau**2)
#         grad = grad + logit / (tau**2)
#         hess = hess + 1 / (tau**2)

#     return nll, grad, hess


# def binomial_mixture_normed_loss(
#     input, target, n, bad_score, tau=0, iters=5, max_step_frac=0.5, reduction="mean"
# ):
#     """
#     Compute the normalized negative log-likelihood (NLL) for the binomial mixture model.

#     Parameters
#     ----------
#     input : torch.Tensor
#         Logit of the predicted probability.
#     target : torch.Tensor
#         Reference counts (number of successes).
#     n : torch.Tensor or float
#         Total read depth (number of trials).
#     bad_score : torch.Tensor or float
#         BAD score for each observation.
#     tau : float, optional
#         Prior scale parameter. If tau > 0, a quadratic prior is added. Default is 0.
#     iters : int, optional
#         Number of Newton-Raphson iterations for root finding (default is 5).
#     max_step_frac : float, optional
#         Fraction of the mean distance from each root to the initial guesses that any
#         single NR step can move. Prevents large jumps. Default: 0.5.
#     reduction : str, optional
#         Specifies the reduction to apply to the output: 'mean' or 'none' (default is 'mean').

#     Returns
#     -------
#     torch.Tensor
#         The normalized negative log-likelihood (NLL) for the input logits. If reduction is 'mean', returns a scalar tensor;
#         otherwise, returns a tensor of losses per element.

#     Notes
#     -----
#     - Finds the minimum NLL by optimizing the logit using Newton-Raphson iterations.
#     - Returns the difference between the NLL at the input and the minimum NLL for normalization.
#     - Useful for comparing model predictions to the best possible prediction for each observation.
#     """

#     d = torch.log(bad_score)

#     # Initial roots: logit of empirical probability ± log_bad_score
#     logit_p = torch.logit(target / n)
#     initial_roots = torch.stack(
#         [logit_p - 2 * d, logit_p - d, logit_p, logit_p + d, logit_p + 2 * d], dim=0
#     )
#     roots = initial_roots

#     # Newton-Raphson optimization to refine roots
#     for _ in range(iters):
#         _, grad, hess = binomial_mixture_nll_grad_hess(roots, target, n, d, tau)
#         max_step = (roots.unsqueeze(0) - initial_roots.unsqueeze(1)).abs().mean(
#             dim=0
#         ) * max_step_frac
#         step = -grad / hess
#         roots += torch.minimum(torch.abs(step), max_step) * torch.sign(step)

#     # Compute NLL at roots and input
#     nll_at_roots, _, _ = binomial_mixture_nll_grad_hess(roots, target, n, d, tau)
#     min_nll = torch.min(nll_at_roots, dim=0)[0]

#     nll_at_input, _, _ = binomial_mixture_nll_grad_hess(input, target, n, d, tau)

#     normalized_nll = nll_at_input - min_nll

#     return normalized_nll.mean() if reduction == "mean" else normalized_nll
