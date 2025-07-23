import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset

from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor

from .utils import one_hot_encode


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
        
        #class is positive,negative gcmatch negative
        #sample_id is agnumber

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
            X_seq = one_hot_encode(dna_seq, dtype=np.float32)
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
            "seq": X_seq.astype(np.float32),                         # np.ndarray (float32)
            "embed": X_embed.astype(np.float32),                     # np.ndarray (float32)
            "indicator": int(indicator),                             # int
            "density": float(density),                               # float
            "r": float(r),                                           # float
            "read_depth": float(read_depth),                         # float
            "chrom": str(chrom, "utf-8") if isinstance(chrom, bytes) else str(chrom),
            "mid": int(mid) if not isinstance(mid, bytes) else int(mid.decode("utf-8")),
            "sample_id": str(sample_id, "utf-8") if isinstance(sample_id, bytes) else str(sample_id),
            "class": str(self.samples["class"][i], "utf-8") if isinstance(self.samples["class"][i], bytes) else str(self.samples["class"][i]),
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
                one_hot_encode(seq, dtype=np.float32)
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
