import numpy as np

import numba


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


def get_iupac_char_from_alleles(alleles):
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
