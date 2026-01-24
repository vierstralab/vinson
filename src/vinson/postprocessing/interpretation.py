import torch
import numpy as np
from tqdm import tqdm
import itertools
from collections.abc import Iterable

from vinson.utils.sequence_utils import force_strict_ohe

from tangermeme.ersatz import dinucleotide_shuffle as dinuc_shuffle
from tangermeme.predict import predict as tangermeme_predict
from tangermeme.product import _apply
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear


def dinucleotide_shuffle(X, **kwargs):
    """
    Shuffle input sequences while preserving dinucleotide composition.

    This function ensures the input tensor is strictly one-hot encoded, then applies
    dinucleotide shuffling to each sequence. Dinucleotide shuffling randomizes the
    sequence order while maintaining the original dinucleotide (adjacent base pair)
    frequencies, which is useful for generating background/control sequences in
    sequence analysis tasks.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (batch, 4, sequence_length), representing one-hot encoded sequences.
    **kwargs
        Additional keyword arguments passed to the underlying dinucleotide shuffling function.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as X, with each sequence dinucleotide-shuffled.
    """
    X_ = force_strict_ohe(X)
    return dinuc_shuffle(X_, **kwargs)


def apply_product(
    func,
    model,
    X,
    product_func=lambda x: (x[0], list(zip(*x[1:]))),
    order=None,
    batch_size=32,
    device="cuda",
    additional_func_kwargs={},
    verbose=False,
    **kwargs,
):
    """Apply a function on the cartesian product between X and each args.

    This function will take the provided function and apply it in a batched
    manner across the cartesian product of `X` and each of the arguments
    provided in `args`. Because this is a cartesian product, the number of
    examples that need to be processed will quickly grow with respect to the
    number of arguments being passed in. Each of the tensors in `args` must
    be one input to `model`, in the order that they are specified by the
    forward function.

    This function can accept in any other function -- be it predictions,
    attributions, or marginalizations. If the provided function itself has
    parameters that need to be specified, you can provide them directly to
    this function in the order that they appear in the provided function.


    Parameters
    ----------
    func: function
        A function, likely implemented in tangermeme, to apply in a batched
        manner across the product of examples.

    model: torch.nn.Module
        The PyTorch model to use to make predictions.

    X: tuple or list
        A tuple or list of model inputs.

    product_func: function
        A function to group the inputs for the product

    batch_size: int, optional
        The number of examples to make predictions for at a time. Default is 32.

    device: str or torch.device
        The device to move the model and batches to when making predictions. If
        set to 'cuda' without a GPU, this function will crash and must be set
        to 'cpu'. Default is 'cuda'.

    additional_func_kwargs: dict, optional
        Additional named arguments to pass into the function when it is called.
        This is provided as an alternate path to route arguments into the
        function in case they overlap, name-wise, with those in this function,
        or if you want to be absolutely sure that the arguments are making
        their way into the function. Default is {}.

    verbose: bool, optional
        Whether to display a progress bar as spacings are evaluated. Default
        is False.

    kwargs: optional
        Additional named arguments that will get passed into the function when
        it is called. Default is no arguments are passed in.


    Returns
    -------
    y: torch.Tensor or list/tuple of torch.Tensors
        The output from the model for each input example. The precise format
        is determined by the model. If the model outputs a single tensor,
        y is a single tensor concatenated across all batches. If the model
        outputs multiple tensors, y is a list of tensors which are each
        concatenated across all batches.
    """

    model = model.to(device).eval()

    n_inputs = len(X)

    prod_args = product_func(X)

    if not order:
        order = list(range(n_inputs))

    y, args_ = [], [[] for _ in range(n_inputs)]

    for x in tqdm(itertools.product(*prod_args), disable=not verbose):
        # Flatten args
        _args = ()
        for arg in x:
            if not isinstance(arg, Iterable):
                arg = (arg,)
            _args += arg

        # Re-order args for model forward pass
        [args_[order[i]].append(a) for i, a in enumerate(_args)]

        if len(args_[0]) == batch_size:
            y_ = _apply(
                func,
                model,
                args_[0],
                args=args_[1:],
                batch_size=batch_size,
                device=device,
                verbose=verbose,
                additional_func_kwargs=additional_func_kwargs,
                **kwargs,
            )
            y.append(y_)

            args_ = [[] for _ in range(n_inputs)]
        else:
            if len(args_[0]) > 0:
                y_ = _apply(
                    func,
                    model,
                    args_[0],
                    args=args_[1:],
                    batch_size=batch_size,
                    device=device,
                    verbose=verbose,
                    additional_func_kwargs=additional_func_kwargs,
                    **kwargs,
                )
                y.append(y_)

    Xal = [len(args) for args in prod_args]

    # If there is only a single output, just concatenate the tensors
    if isinstance(y[0], torch.Tensor):
        yl = y[0].shape[1:]
        y = torch.cat(y).reshape(*Xal, *yl)
    else:
        _y = []

        # If either the function or the model have multiple outputs, but the
        # other has a single output, then concatenate tensors across the
        # outputs appropriately.
        if isinstance(y[0][0], torch.Tensor):
            for y_ in list(zip(*y)):
                yl = y_[0].shape[1:]
                _y.append(torch.cat(y_).reshape(*Xal, *yl))

        # If both the function and the model have multiple outputs then you
        # have to go one layer deeper when concatenating the tensors.
        else:
            for y_task in list(zip(*y)):
                _y.append([])

                for y_ in list(zip(*y_task)):
                    yl = y_[0].shape[1:]
                    _y[-1].append(torch.cat(y_).reshape(*Xal, *yl))

        y = _y

    return y


class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.exp = _Exp()

    def forward(self, seq, embed):
        x = self.model(seq, embed)
        if self.model.log_output:
            x = self.exp(x)
        return x
    
    def get_sequence_attributions(self, X, X_embed, print_convergence_deltas=True, **kwargs):
        attributions = deep_lift_shap(
            self,
            X,
            args=(X_embed,),
            device="cuda" if torch.cuda.is_available() else "cpu",
            print_convergence_deltas=print_convergence_deltas,
            references=dinucleotide_shuffle,
            additional_nonlinear_ops={
                _Exp: _nonlinear
            },
            **kwargs,
        )
        return attributions
