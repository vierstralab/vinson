import torch

import numpy as np
import numba

from genome_tools import GenomicInterval
from collections.abc import Iterable



IUPAC_DNA = "XACMGRSVTWYHKDBN"

_dna_embed = np.zeros((256, 4), dtype=np.int8)
_dna_embed[ord("A")] = np.array([1, 0, 0, 0])
_dna_embed[ord("C")] = np.array([0, 1, 0, 0])
_dna_embed[ord("G")] = np.array([0, 0, 1, 0])
_dna_embed[ord("T")] = np.array([0, 0, 0, 1])
_dna_embed[ord("W")] = np.array([1, 0, 0, 1])
_dna_embed[ord("S")] = np.array([0, 1, 1, 0])
_dna_embed[ord("M")] = np.array([1, 1, 0, 0])
_dna_embed[ord("K")] = np.array([0, 0, 1, 1])
_dna_embed[ord("R")] = np.array([1, 0, 1, 0])
_dna_embed[ord("Y")] = np.array([0, 1, 0, 1])
_dna_embed[ord("B")] = np.array([0, 1, 1, 1])
_dna_embed[ord("D")] = np.array([1, 0, 1, 1])
_dna_embed[ord("H")] = np.array([1, 1, 0, 1])
_dna_embed[ord("V")] = np.array([1, 1, 1, 0])
_dna_embed[ord("N")] = np.array([1, 1, 1, 1])


@numba.njit("void(int8[:, :], int8[:], int8[:, :])", cache=True)
def _fast_one_hot_encode(X_ohe, seq, mapping):
    """Fast encoding of characters with a reference embedding

    Parameters
    ----------
    X_ohe : numpy.ndarray
        The one-hot encoding matrix of shape (sequence_length, 4),
        of type int8
    seq : numpy.ndarray
        Array of utf-8 encoded characters of type int8
    mapping : numpy.ndarray
        Array of mapping where indicies are utf-8 encoded characters
        of shape (256, 4) where the the first the indicies correspond
        to the UTF-8 encoding of a character, of type int8
    """
    for i in range(len(seq)):
        base_embed = mapping[seq[i]]
        X_ohe[i, :] = base_embed


def one_hot_encode(sequence, dtype=np.float32):
    """_Converts a string or list of characters into a one-hot encoding.

    Parameters
    ----------
    sequence : str or list
        The sequence to convert to a one-hot encoding.
    dtype : str or torch.dtype, optional
        _description_, by default np.float32

    Returns
    -------
    ohe: numpy.ndarray
        A matrix of shape (4, sequence_length)
    """
    seq_idxs = np.frombuffer(bytearray(sequence, "utf8"), dtype=np.int8)
    n, m = len(sequence), _dna_embed.shape[1]
    
    one_hot_encoding = np.zeros((n, m), dtype=np.int8)

    _fast_one_hot_encode(one_hot_encoding, seq_idxs, _dna_embed)

    one_hot_encoding = one_hot_encoding.astype(dtype)
    one_hot_encoding /= one_hot_encoding.sum(axis=1)[:, np.newaxis]

    return one_hot_encoding.T


def force_strict_ohe(x):
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
    x_ = x.clone()
    seq_idx, pos = torch.where(torch.max(x, dim=1)[0] < 1.0)
    for i, j in zip(seq_idx, pos):
        base = np.random.choice(4, p=x[i, :, j])
        x_[i, :, j] = torch.zeros(4)
        x_[i, base, j] = 1.0
    return x_

def get_iupac_char_from_alleles(*alleles):
    """Returns the IUPAC character from a list of possible
    DNA nucleotides.

    Parameters
    ----------
    alleles : list
        A list of DNA characters

    Returns
    -------
    iupac : str
        The list of alleles encoded in a IUPAC DNA
        base identity
    """

    i = None
    for allele in alleles:
        j = IUPAC_DNA.find(allele)
        if i is None:
            i = j
        else:
            i = i | j
    return IUPAC_DNA[i]


def intervals_to_ohe(intervals, seqlen, fasta_extr):
    """
    Convert genomic intervals to one-hot encoded DNA sequences.

    For each interval, extracts the sequence from the reference genome using the provided
    FastaExtractor, centers the interval, and one-hot encodes the sequence.

    Parameters
    ----------
    intervals : Iterable[GenomicInterval] or GenomicInterval
        List of GenomicInterval objects or a single GenomicInterval.
    seqlen : int
        Length of the sequence window to extract and encode.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.

    Returns
    -------
    np.ndarray
        Array of shape (N, 4, seqlen), where N is the number of intervals, containing
        one-hot encoded DNA sequences.
    """
    ohe = []

    if not isinstance(intervals, Iterable):
        intervals = [intervals]

    for i in intervals:
        mid = (i.start + i.end) // 2
        seq = fasta_extr[GenomicInterval(i.chrom, mid, mid).widen(seqlen//2)]
        ohe.append(one_hot_encode(seq.upper()))
    
    return np.stack(ohe)

def variants_to_ohe(variants, seqlen, fasta_extr):
    """
    Convert variant objects to one-hot encoded reference and alternate allele sequences.

    For each variant, extracts the reference sequence from the genome, creates the alternate
    sequence by substituting the alternate allele, and one-hot encodes both.

    Parameters
    ----------
    variants : Iterable or object
        List of variant objects or a single variant object. Each variant must have
        'chrom', 'start', and 'alt' attributes.
    seqlen : int
        Length of the sequence window to extract and encode.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.

    Returns
    -------
    np.ndarray
        Array of shape (2*N, 4, seqlen), where N is the number of variants, containing
        one-hot encoded reference and alternate allele sequences for each variant.
    """
    mid = seqlen // 2
    ohe = []

    if not isinstance(variants, Iterable):
        variants = [variants]

    for v in variants:
        # reference 
        seq_ref = fasta_extr[GenomicInterval(v.chrom, v.start, v.start).widen(mid)]
        ohe.append(one_hot_encode(seq_ref))
        # alternate
        seq_alt = seq_ref[:mid] + v.alt + seq_ref[mid+1:]
        ohe.append(one_hot_encode(seq_alt))

    return np.stack(ohe)
