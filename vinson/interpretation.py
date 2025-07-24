import torch

from tangermeme.ersatz import dinucleotide_shuffle

def dinucleotide_shuffle_variant(X_ref, X_alt, n=20, random_state=None):
    """Shuffles set one sequences with SNVs. 
    
    This performs dinucleotide shuffling and then places the SNVs exactly
    as they were (same location and identity).
    
    Parameters
    ----------
    X_ref : torch.tensor, shape=(-1, len(alphabet), length)
        A one-hot encoded set of sequences to be shuffled
        containing the REF alleles.
    X_alt : torch.tensor, shape=(-1, len(alphabet), length)
        A one-hot encoded set of sequences to be shuffled
        containing the ALT alleles.
    n : int, optional
        Number of shuffles to perform, by default 20
    random_state : _type_, optional
        Random number generator seed, by default None

    Returns
    -------
    shuffled_sequences: (torch.tensor, torch.tensor), each shape (-1, n, k, -1) 
        The shuffled sequences.
    """
    diff = (X_ref - X_alt).abs()
    pos = torch.where(diff.sum(dim=1) > 0)

    assert len(pos[0]) > 0, "Inputs are the exact same!"

    X_ref_shuf = dinucleotide_shuffle(X_ref, n=n, random_state=random_state)
    X_alt_shuf = torch.clone(X_ref_shuf)

    for seq_idx, k in zip(*pos):
        X_ref_shuf[seq_idx, :, :, k] = X_ref[seq_idx][:, k].repeat(n, 1)
        X_alt_shuf[seq_idx, :, :, k] = X_alt[seq_idx][:, k].repeat(n, 1)
    
    return X_ref_shuf, X_alt_shuf

