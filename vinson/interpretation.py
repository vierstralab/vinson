import numpy as np
import torch

from tangermeme.ersatz import dinucleotide_shuffle as dinuc_shuffle


def force_strict_ohe(X):
    """
    Ensure input tensor is strictly one-hot encoded along the nucleotide axis.

    For each position in each sequence, if the maximum value along the nucleotide axis is less than 1.0,
    randomly selects a base according to the current probabilities and sets that base to 1.0 (others to 0).
    Returns a tensor with strict one-hot encoding at every position.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (batch, 4, sequence_length), representing probabilistic one-hot encoding.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as X, with strict one-hot encoding at every position.
    """
    X_ = X.clone()
    seq_idx, pos = torch.where(torch.max(X, dim=1)[0] < 1.0)
    for i, j in zip(seq_idx, pos):
        base = np.random.choice(4, p=X[i, :, j])
        X_[i, :, j] = torch.zeros(4)
        X_[i, base, j] = 1.0
    return X_


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
