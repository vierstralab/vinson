import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset

from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

from .utils import one_hot_encode, get_iupac_char_from_alleles

import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        samples_file,
        embeddings_file,
        fasta_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
        random_state=None,
        seqlen=1344,
    ):
        self.fasta_file = fasta_file
        self.samples_file = samples_file
        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise

        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.random_state = np.random.RandomState(random_state)

        self.fasta_extr = None

        logger.info("Opening samples file.")
        self.samples = h5py.File(samples_file, "r")

        logger.info("Loading embeddings.")
        self.embeddings_df = pd.read_table(embeddings_file, index_col=0)

    def __del__(self):
        if self.samples:
            self.samples.close()

        if self.fasta_extr:
            self.fasta_extr.close()

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class SequenceEmbeddingDataset(BaseDataset):
    """
    PyTorch Dataset for extracting sequence and cell-type embeddings, with optional
    genotype injection and read depth normalization.

    Parameters
    ----------
    samples_file : str
        Path to HDF5 file containing sample data and labels.
    embeddings_file : str
        Path to tab-delimited file with cell-type/state embeddings.
    read_depth_file : str
        Path to tab-delimited file with sample read depths.
    fasta_file : str
        Path to reference genome FASTA file.
    sample_genotype_file : str, optional
        Path to genotype metadata file (tab-delimited).
    genotype_file : str, optional
        Path to genotype file in tabix format.
    reverse_complement : bool, optional
        If True, randomly reverse-complement sequences for augmentation (default: True).
    jitter : int, optional
        Maximum number of bases to randomly shift the region (default: 0).
    noise : float, optional
        Standard deviation of Gaussian noise added to embeddings (default: 0).
    random_state : int or None, optional
        Seed for random number generator (default: None).
    random_sample : bool, optional
        If True, randomly sample cell-type embeddings (default: False).

    Attributes
    ----------
    samples : h5py.File
        Opened HDF5 file with sample data.
    embeddings_df : pandas.DataFrame
        DataFrame of cell-type/state embeddings.
    read_depths : pandas.Series
        Series of sample read depths.
    sample_to_genotype_df: pandas.DataFrame
        DataFrame of genotype metadata.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.
    genotype_extr : TabixExtractor
        Extractor for genotype data.

    Notes
    -----
    - Injects genotypes into reference sequence if genotype_file is provided.
    - Supports region jittering and reverse complementation for data augmentation.
    - Returns a dictionary with sequence, embedding, indicator, density, dispersion,
      read depth, and class label for each sample.
    """

    def __init__(
        self,
        samples_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        sample_genotype_file=None,
        genotype_file=None,
        reverse_complement=True,
        jitter=0,
        noise=0,
        random_state=None,
    ):
        super(SequenceEmbeddingDataset, self).__init__(
            samples_file,
            embeddings_file,
            fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            random_state=random_state,
        )

        self.genotype_file = genotype_file
        self.genotype_extr = None

        assert set(["chrom", "mid", "class", "disp", "density", "sample_id"]).issubset(
            self.samples.keys()
        )

        logger.info("Loading sample read depths.")
        self.read_depths = pd.read_table(read_depth_file, index_col=0).iloc[:, 0]

        if sample_genotype_file:
            logger.info("Loading genotype metadata...")
            self.sample_to_genotype_df = pd.read_table(
                sample_genotype_file, index_col=0
            )
            # TODO: check if genotype file is available and readable
            self.include_genotypes = True
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )
            self.include_genotypes = False

    def __getitem__(self, i):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)
        # tabix is not thread-safe
        if self.include_genotypes and not self.genotype_extr:
            self.genotype_extr = TabixExtractor(self.genotype_file)

        chrom, mid, sample_id, indicator, density, r = (
            self.samples["chrom"][i].astype(str),
            self.samples["mid"][i],
            self.samples["sample_id"][i].astype(str),
            1 if self.samples["class"][i].astype(str) == "positive" else 0,
            self.samples["density"][i],
            self.samples["disp"][i],
        )

        # Define region
        interval = GenomicInterval(chrom, mid, mid).widen(self.seqlen//2)

        # Jitter/shift region as necesary
        if self.jitter > 0:
            shift = self.random_state.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        # Get DNA sequence
        dna_seq = self.fasta_extr[interval]

        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            try:
                indiv_id = self.sample_to_genotype_df.loc[sample_id].indiv_id
                variants = self.genotype_extr[interval]
                variants = variants[variants[13].str.contains(indiv_id)]
            
                logger.debug(
                    f"Found {len(variants)} variants in sample {sample_id} from individual {indiv_id}"
                )

                for i, v in variants.iterrows():
                    try:
                        pos, ref, alt, gt = int(v[1]) - interval.start, v[4], v[5], v[8]
                    except TypeError as e:
                        logger.error(f"Error parsing variant: {v}")
                        raise e

                    # Get IUPAC base charater
                    if gt == "0/1" or gt == "1/0":
                        base = get_iupac_char_from_alleles((ref, alt))
                    elif gt == "1/1":
                        base = alt
                    else:
                        continue  # if reference do nothing

                    dna_seq = dna_seq[:pos] + base + dna_seq[pos + 1 :]

            except Exception as e:
                logger.debug(f"Error: {sample_id} -- {indiv_id}")
                # pass
            
        # One-hot encode DNA sequence
        try:
            X_seq = one_hot_encode(dna_seq, dtype=np.float32)
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{mid} -- {dna_seq})"
            )
            raise e

        # Reverse complete (augmentation)
        if self.reverse_complement and self.random_state.choice(2) == 1:
            X_seq = np.flip(X_seq, [0, 1])

        # Cell type/state embeddings
        X_embed = self.embeddings_df[sample_id].to_numpy(dtype=np.float32)

        # Add a little Gaussian noise to embeddings
        if self.noise > 0:
            X_embed = X_embed + self.random_state.normal(
                0, self.noise, len(X_embed)
            ).astype(np.float32)

        # Sample read depth
        read_depth = self.read_depths.loc[sample_id]

        return {
            "seq": X_seq.copy(),
            "embed": X_embed.copy(),
            "indicator": indicator,
            "density": density if density < 10.0 else 10.0,
            "r": r,
            "read_depth": read_depth,
            "class": self.samples["class"][i].astype(str),
        }

    def __len__(self):
        return self.samples["chrom"].shape[0]


class VariantEmbeddingDataset(BaseDataset):
    """
    PyTorch Dataset for extracting reference and alternate allele sequences and
    cell-type embeddings for variant effect prediction.

    Parameters
    ----------
    samples_file : str
        Path to HDF5 file containing variant metadata and counts.
    embeddings_file : str
        Path to tab-delimited file with cell-type/state embeddings.
    fasta_file : str
        Path to reference genome FASTA file.
    reverse_complement : bool, optional
        If True, randomly reverse-complement sequences for augmentation (default: True).
    jitter : int, optional
        Maximum number of bases to randomly shift the region (default: 0).
    noise : float, optional
        Standard deviation of Gaussian noise added to embeddings (default: 0).
    random_state : int or None, optional
        Seed for random number generator (default: None).

    Attributes
    ----------
    samples : h5py.File
        Opened HDF5 file with variant metadata and counts.
    embeddings_df : pandas.DataFrame
        DataFrame of cell-type/state embeddings.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.

    Notes
    -----
    - Extracts both reference and alternate allele sequences for each variant.
    - Supports region jittering and reverse complementation for data augmentation.
    - Returns a dictionary with reference and alternate sequences, embedding,
      reference and total counts, BAD score, and log fold change for each variant.
    """

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
        super(VariantEmbeddingDataset, self).__init__(
            samples_file,
            embeddings_file,
            fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            random_state=random_state
        )

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

    def __getitem__(self, i):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

        chrom, pos, ref, alt, ref_counts, total_counts, bad, lfc, sample_id = (
            self.samples["chrom"][i].astype(str),
            self.samples["pos"][i],
            self.samples["ref"][i].astype(str),
            self.samples["alt"][i].astype(str),
            self.samples["ref_counts"][i],
            self.samples["total_counts"][i],
            self.samples["BAD"][i],
            self.samples["logit_es"][i],
            self.samples["sample_id"][i].astype(str),
        )

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen//2)

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
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{variant.start})"
            )
            raise e

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X_seq[0] = np.flip(X_seq[0], [0, 1])
            X_seq[1] = np.flip(X_seq[1], [0, 1])

        # Cell type embeddings
        X_embed = self.embeddings_df[sample_id].to_numpy(dtype=np.float32)

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
            "lfc": lfc / np.log(2),
        }

    def __len__(self):
        return self.samples["chrom"].shape[0]
