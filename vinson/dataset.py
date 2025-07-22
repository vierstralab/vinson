import numpy as np
import pandas as pd
import numba

import h5py

import torch
from torch.utils.data import Dataset

from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor

@numba.njit("void(int8[:, :], int8[:], int8[:])", cache=True)
def _fast_one_hot_encode(X_ohe, seq, mapping):
    """An internal function for quickly converting bytes to one-hot indexes."""

    for i in range(len(seq)):
        idx = mapping[seq[i]]
        if idx == -1:
            continue

        if idx == -2:
            raise ValueError(
                "Encountered character that is not in " + "`alphabet` or in `ignore`."
            )

        X_ohe[i, idx] = 1


def one_hot_encode(
    sequence,
    alphabet=["A", "C", "G", "T"],
    dtype=float,
    ignore=["N"],
    desc=None,
    verbose=False,
    **kwargs,
):
    """Converts a string or list of characters into a one-hot encoding.

    This function will take in either a string or a list and convert it into a
    one-hot encoding. If the input is a string, each character is assumed to be
    a different symbol, e.g. 'ACGT' is assumed to be a sequence of four
    characters. If the input is a list, the elements can be any size.

    Although this function will be used here primarily to convert nucleotide
    sequences into one-hot encoding with an alphabet of size 4, in principle
    this function can be used for any types of sequences.

    Parameters
    ----------
    sequence : str or list
            The sequence to convert to a one-hot encoding.

    alphabet : set or tuple or list
            A pre-defined alphabet where the ordering of the symbols is the same
            as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
            the returned tensor will have a 1 at index 0 if the character was 'A'.
            Characters outside the alphabet are ignored and none of the indexes are
            set to 1. Default is ['A', 'C', 'G', 'T'].

    dtype : str or torch.dtype, optional
            The data type of the returned encoding. Default is int8.

    ignore: list, optional
            A list of characters to ignore in the sequence, meaning that no bits
            are set to 1 in the returned one-hot encoding. Put another way, the
            sum across characters is equal to 1 for all positions except those
            where the original sequence is in this list. Default is ['N'].


    Returns
    -------
    ohe : numpy.ndarray
            A binary matrix of shape (alphabet_size, sequence_length) where
            alphabet_size is the number of unique elements in the sequence and
            sequence_length is the length of the input sequence.
    """

    for char in ignore:
        if char in alphabet:
            raise ValueError(
                "Character {} in the alphabet ".format(char)
                + "and also in the list of ignored characters."
            )

    if isinstance(alphabet, list):
        alphabet = "".join(alphabet)

    ignore = "".join(ignore)

    e = "utf8"
    seq_idxs = np.frombuffer(bytearray(sequence, e), dtype=np.int8)
    alpha_idxs = np.frombuffer(bytearray(alphabet, e), dtype=np.int8)
    ignore_idxs = np.frombuffer(bytearray(ignore, e), dtype=np.int8)

    one_hot_mapping = np.zeros(256, dtype=np.int8) - 2
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i

    for i, idx in enumerate(ignore_idxs):
        one_hot_mapping[idx] = -1

    n, m = len(sequence), len(alphabet)

    one_hot_encoding = np.zeros((n, m), dtype=np.int8)
    _fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)
    return one_hot_encoding.astype(dtype).T


class SequenceEmbeddingDataset(Dataset):
    def __init__(
        self,
        samples_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        reverse_complement=True,
        jitter=0,
        noise=0,
        random_state=None,
        random_sample=False,
    ):
        self.fasta_file = fasta_file
        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise
        self.random_sample = random_sample

        self.random_state = np.random.RandomState(random_state)

        self.fasta_extr = None
        self.samples = None

        print("Opening samples file...")
        self.samples = h5py.File(samples_file, "r")

        assert set(["chrom", "mid", "class", "disp", "density", "sample_id"]).issubset(
            self.samples.keys()
        )

        print("Loading embeddings...")
        self.embeddings = pd.read_table(embeddings_file, index_col=0)

        print("Loading sample read depths...")
        self.read_depths = pd.read_table(read_depth_file, index_col=0).iloc[:, 0]

        print("Done!")

    def __del__(self):
        if self.samples:
            self.samples.close()

    def __getitem__(self, i):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

        chrom, mid, sample_id, indicator, density, r = (
            self.samples["chrom"][i].astype(str),
            self.samples["mid"][i],
            self.samples["sample_id"][i].astype(str),
            1 if self.samples["class"][i].astype(str) == "positive" else 0,
            self.samples["density"][i],
            self.samples["disp"][i],
        )

        interval = GenomicInterval(chrom, mid, mid).widen(672)

        if self.jitter > 0:
            shift = self.random_state.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        dna_seq = self.fasta_extr[interval]

        try:
            X_seq = one_hot_encode(
                dna_seq,
                ignore=["W", "S", "M", "K", "R", "Y", "B", "D", "H", "V", "N"],
                dtype=np.float32,
            )
        except ValueError as e:
            print(
                f"Error converting DNA to one-hot encoding ({chrom}:{mid} -- {dna_seq})"
            )
            raise e

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X_seq = np.flip(X_seq, [0, 1])

        # Cell type embeddings
        if self.random_sample:
            sample_idx = self.random_state.choice(self.embeddings.shape[1])
            X_embed = self.embeddings.iloc[:, sample_idx].to_numpy(dtype=np.float32)
        else:
            X_embed = self.embeddings[sample_id].to_numpy(dtype=np.float32)

        if self.noise > 0:
            X_embed = X_embed + self.random_state.normal(
                0, self.noise, len(X_embed)
            ).astype(np.float32)

        # Parameters
        read_depth = self.read_depths.loc[sample_id]

        return {
            "seq": X_seq.copy(),
            "embed": X_embed.copy(),
            "indicator": indicator,
            "density": density if density < 10.0 else 10.0,
            "r": r,
            "read_depth": read_depth,
        }

    def __len__(self):
        return self.samples["chrom"].shape[0]


class VariantEmbeddingDataset(Dataset):
    def __init__(
        self,
        samples_file,
        embeddings_file,
        fasta_file,
        reverse_complement=True,
        jitter=0,
        noise=0,
        random_state=None,
    ):
        self.fasta_file = fasta_file
        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise

        self.random_state = np.random.RandomState(random_state)

        self.fasta_extr = None
        self.samples = None

        print("Opening samples file...")
        self.samples = h5py.File(samples_file, "r")

        assert set(
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "ref_counts",
                "total_counts",
                "BAD",
                "sample_id",
            ]
        ).issubset(self.samples.keys())

        print("Loading embeddings...")
        self.embeddings = pd.read_table(embeddings_file, index_col=0)

        print("Done!")

    def __del__(self):
        if self.samples:
            self.samples.close()

    def __getitem__(self, i):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

        chrom, pos, ref, alt, ref_counts, total_counts, bad, sample_id = (
            self.samples["chrom"][i].astype(str),
            self.samples["pos"][i],
            self.samples["ref"][i].astype(str),
            self.samples["alt"][i].astype(str),
            self.samples["ref_counts"][i],
            self.samples["total_counts"][i],
            self.samples["BAD"][i],
            self.samples["sample_id"][i].astype(str),
        )

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(672)

        if self.jitter > 0:
            shift = self.random_state.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        pos = variant.start - interval.start

        dna_seq_ref = self.fasta_extr[interval]
        dna_seq_alt = dna_seq_ref[:pos] + alt + dna_seq_ref[pos + 1 :]

        try:
            X_seq = [
                one_hot_encode(
                    seq,
                    ignore=["W", "S", "M", "K", "R", "Y", "B", "D", "H", "V", "N"],
                    dtype=np.float32,
                )
                for seq in [dna_seq_ref, dna_seq_alt]
            ]
        except ValueError as e:
            print(f"Error converting DNA to one-hot encoding ({chrom}:{variant.start})")
            raise e

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X_seq[0] = np.flip(X_seq[0], [0, 1])
            X_seq[1] = np.flip(X_seq[1], [0, 1])

        # Cell type embeddings
        X_embed = self.embeddings[sample_id].to_numpy(dtype=np.float32)

        if self.noise > 0:
            X_embed = X_embed + self.random_state.normal(
                0, self.noise, len(X_embed)
            ).astype(np.float32)

        return {
            "seq_ref": X_seq[0].copy(),
            "seq_alt": X_seq[1].copy(),
            "embed": X_embed.copy(),
            "ref_counts": ref_counts,
            "total_counts": total_counts,
            "bad_score": bad,
        }

    def __len__(self):
        return self.samples["chrom"].shape[0]
