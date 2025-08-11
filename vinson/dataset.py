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
    """
    Base PyTorch Dataset for genomic sequence and embedding data.

    Provides common initialization and resource management for derived datasets,
    including loading sample metadata, embeddings, and setting up random state.

    Parameters
    ----------
    samples_file : str
        Path to HDF5 file containing sample data.
    embeddings_file : str
        Path to tab-delimited file with cell-type/state embeddings.
    fasta_file : str
        Path to reference genome FASTA file.
    reverse_complement : bool, optional
        If True, randomly reverse-complement sequences for augmentation (default: False).
    jitter : int, optional
        Maximum number of bases to randomly shift the region (default: 0).
    noise : float, optional
        Standard deviation of Gaussian noise added to embeddings (default: 0).
    seed : int or None, optional
        Seed for random number generator (default: None).
    seqlen : int, optional
        Length of the sequence window (default: 1344, must be even).

    Attributes
    ----------
    samples : h5py.File
        Opened HDF5 file with sample data.
    embeddings_df : pandas.DataFrame
        DataFrame of cell-type/state embeddings.
    fasta_extr : FastaExtractor or None
        Extractor for reference genome sequences (initialized as None).
    random_state : np.random.RandomState
        Random number generator for reproducibility.
    seqlen : int
        Length of the sequence window.
    """
    def __init__(
        self,
        samples_file,
        embeddings_file,
        fasta_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
        seed=None,
        seqlen=1344,
    ):
        self.fasta_file = fasta_file
        self.samples_file = samples_file
        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise

        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.fasta_extr = None

        logger.info("Opening samples file.")
        self.samples = h5py.File(samples_file, "r")

        logger.info("Loading embeddings.")
        self.embeddings_df = pd.read_table(embeddings_file, index_col=0)

        self.seed = seed
        self.reset_random_state()

    def __del__(self):
        """
        Clean up open file handles for samples and FASTA extractor.
        """
        if self.samples:
            self.samples.close()

        if self.fasta_extr:
            self.fasta_extr.close()

    def __getitem__(self, i):
        """
        Retrieve a single data item by index.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def reset_random_state(self):
        """
        Reset the random number generator using the stored seed.
        """
        self.random_state = np.random.RandomState(self.seed)


class SeqEmbedDataset(BaseDataset):
    """
    PyTorch Dataset for extracting sequence and cell-type embeddings, with optional
    genotype injection, negative sampling, and read depth normalization.

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
    negative_samples_file : str, optional
        Path to tabix file containing negative samples for augmentation.
    negative_samples_rate : int, optional
        Ratio of negative to positive samples (default: 1).
    negative_samples_weight : float, optional
        Weight assigned to negative samples (default: 2.5).
    sample_genotype_file : str, optional
        Path to genotype metadata file (tab-delimited).
    genotype_file : str, optional
        Path to genotype file in tabix format.
    clip_density : float, optional
        Maximum allowed density value (default: 5).
    min_bg : float, optional
        Minimum allowed background value (default: 0.05).
    reverse_complement : bool, optional
        If True, randomly reverse-complement sequences for augmentation (default: False).
    jitter : int, optional
        Maximum number of bases to randomly shift the region (default: 0).
    noise : float, optional
        Standard deviation of Gaussian noise added to embeddings (default: 0).
    seed : int or None, optional
        Seed for random number generator (default: None).

    Attributes
    ----------
    samples : h5py.File
        Opened HDF5 file with sample data.
    embeddings_df : pandas.DataFrame
        DataFrame of cell-type/state embeddings.
    read_depths : pandas.Series
        Series of sample read depths.
    sample_to_genotype_df : pandas.DataFrame
        DataFrame of genotype metadata.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.
    genotype_extr : TabixExtractor
        Extractor for genotype data.
    negative_samples_extr : TabixExtractor
        Extractor for negative samples.

    Notes
    -----
    - Injects genotypes into reference sequence if genotype files are provided.
    - Supports region jittering and reverse complementation for data augmentation.
    - Supports negative sampling for training with imbalanced data.
    - Returns a dictionary with sequence, embedding, indicator, density, background,
      read depth, weight, chromosome, midpoint, and sample ID for each sample.
    """

    def __init__(
        self,
        samples_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        negative_samples_file=None,
        negative_samples_rate=1,
        negative_samples_weight=2.5,
        sample_genotype_file=None,
        genotype_file=None,
        clip_density=5,
        min_bg=0.05,
        reverse_complement=False,
        jitter=0,
        noise=0,
        seed=None,
    ):
        super(SeqEmbedDataset, self).__init__(
            samples_file,
            embeddings_file,
            fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            seed=seed,
        )

        self.negative_samples_file = negative_samples_file
        self.negative_samples_rate = negative_samples_rate
        self.negative_samples_weight = negative_samples_weight
        self.negative_samples_extr = None

        self.genotype_file = genotype_file
        self.genotype_extr = None

        self.clip_density = clip_density
        self.min_bg = min_bg

        assert set(["chrom", "mid", "class", "density", "sample_id"]).issubset(
            self.samples.keys()
        )

        logger.info("Loading sample read depths.")
        self.read_depths = pd.read_table(read_depth_file, index_col=0).iloc[:, 0]

        if negative_samples_file:
            logger.info("Sampling from negative examples file...")
            self.sample_from_negatives = True
        else:
            self.sample_from_negatives = False

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

    def get_indiv_sequence(self, interval, sample_id):
        """
        Retrieve the DNA sequence for a given interval and sample, injecting sample-specific
        genotypes if available.

        Parameters
        ----------
        interval : GenomicInterval
            Genomic interval to extract.
        sample_id : str
            Sample identifier.

        Returns
        -------
        int
            Number of variants injected.
        str
            DNA sequence with genotypes injected.
        """
        dna_seq = self.fasta_extr[interval]

        # Get individual ID
        indiv_id = self.sample_to_genotype_df.loc[sample_id].indiv_id
        # Extract variants pertaining to individual
        variants = self.genotype_extr[interval]
        # TODO: Change this to df.query vs. str.contains?
        variants = variants[variants.indiv_id.str.contains(indiv_id)].set_index("start")

        logger.debug(
            f"Found {len(variants)} variants in sample {sample_id} from individual {indiv_id}"
        )

        for v in variants.itertuples():
            # Variant position is the dataframe index
            pos = v.Index
            rel_pos = pos - interval.start

            # Get IUPAC base charater
            if v.gt == "0/1" or v.gt == "1/0":
                base = get_iupac_char_from_alleles((v.ref, v.alt))
            elif v.gt == "1/1":
                base = v.alt
            else:
                base = v.ref  # if reference do nothing

            dna_seq = dna_seq[:rel_pos] + base + dna_seq[rel_pos + 1 :]

        return len(variants), dna_seq

    def __getitem__(self, i):
        """
        Retrieve a single training sample, including sequence, embedding, and metadata.

        Handles negative sampling, region jittering, reverse complementation, genotype injection,
        one-hot encoding, and embedding noise.

        Parameters
        ----------
        i : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Dictionary with keys:
                - 'seq': np.ndarray, one-hot encoded DNA sequence
                - 'embed': np.ndarray, cell-type/state embedding
                - 'indicator': int, 1 for positive, 0 for negative sample
                - 'density': float, normalized density value
                - 'bg': float, background value
                - 'read_depth': float, sample read depth
                - 'weight': float, sample weight
                - 'chrom': str, chromosome name
                - 'mid': int, midpoint coordinate
                - 'sample_id': str, sample identifier
        """
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)
            # tabix is not thread-safe
        if self.sample_from_negatives and not self.negative_samples_extr:
            self.negative_samples_extr = TabixExtractor(self.negative_samples_file)
        # TODO: move this to "get_indiv_sequence" function?
        if self.include_genotypes and not self.genotype_extr:
            self.genotype_extr = TabixExtractor(self.genotype_file)

        idx = i // (self.negative_samples_rate + 1)

        chrom, mid, sample_id, density, bg, indicator = (
            self.samples["chrom"][idx].astype(str),
            self.samples["mid"][idx].astype(int),
            self.samples["sample_id"][idx].astype(str),
            self.samples["density"][idx].astype(np.float32),
            self.samples["bg_mu"][idx].astype(np.float32),
            1 if self.samples["class"][idx].astype(str) == "positive" else 0,
        )

        if self.sample_from_negatives and i % (self.negative_samples_rate + 1):
            try:
                negative_interval = GenomicInterval(chrom, mid, mid + 1)

                negative_sample = (
                    self.negative_samples_extr[negative_interval].sample(
                        n=1, random_state=self.random_state
                    )
                ).values[0, :]

                sample_id = str(negative_sample[3])
                density = np.float32(negative_sample[4])
                bg = np.float32(negative_sample[5])
                indicator = 0

            except ValueError:
                pass

        # Define region
        interval = GenomicInterval(chrom, mid, mid).widen(self.seqlen // 2)

        # Jitter/shift region as necesary
        if self.jitter > 0:
            shift = self.random_state.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            _, dna_seq = self.get_indiv_sequence(interval, sample_id)
        else:
            dna_seq = self.fasta_extr[interval]

        # One-hot encode DNA sequence
        try:
            X_seq = one_hot_encode(dna_seq, dtype=np.float32)
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{mid} -- {sample_id})"
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

        # Adjust values as needed
        density = density if density < self.clip_density else self.clip_density
        weight = 1.0 if indicator else self.negative_samples_weight
        bg = np.nanmax([bg, self.min_bg])

        return {
            "seq": X_seq.copy(),
            "embed": X_embed.copy(),
            "indicator": indicator,
            "density": np.float32(density),
            "bg": np.float32(bg),
            "read_depth": np.float32(read_depth),
            "weight": np.float32(weight),
            "chrom": chrom,
            "mid": mid,
            "sample_id": sample_id,
        }

    def __len__(self):
        """
        Return the number of samples in the dataset, accounting for negative sampling.

        Returns
        -------
        int
            Number of samples (including negatives if enabled).
        """
        N = self.samples["chrom"].shape[0]
        return N * (self.negative_samples_rate + 1) if self.sample_from_negatives else N

    def __del__(self):
        """
        Clean up open file handles for genotype and negative sample extractors.
        """
        super(SeqEmbedDataset, self).__del__()

        if self.genotype_extr:
            self.genotype_extr.close()

        if self.negative_samples_extr:
            self.negative_samples_extr.close()


class VariantEmbedDataset(BaseDataset):
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
    sample_genotype_file : str, optional
        Path to genotype metadata file (tab-delimited).
    genotype_file : str, optional
        Path to genotype file in tabix format.
    flip_alleles : bool, optional
        If True, randomly flip reference and alternate alleles for regularization (default: True).
    reverse_complement : bool, optional
        If True, randomly reverse-complement sequences for augmentation (default: True).
    jitter : int, optional
        Maximum number of bases to randomly shift the region (default: 0).
    noise : float, optional
        Standard deviation of Gaussian noise added to embeddings (default: 0).
    seed : int or None, optional
        Seed for random number generator (default: None).

    Attributes
    ----------
    samples : h5py.File
        Opened HDF5 file with variant metadata and counts.
    embeddings_df : pandas.DataFrame
        DataFrame of cell-type/state embeddings.
    fasta_extr : FastaExtractor
        Extractor for reference genome sequences.
    genotype_extr : TabixExtractor
        Extractor for genotype data.

    Notes
    -----
    - Extracts both reference and alternate allele sequences for each variant.
    - Supports region jittering and reverse complementation for data augmentation.
    - Returns a dictionary with reference and alternate sequences, embedding,
      reference and total counts, BAD score, log fold change, sample ID, and weight.
    """


    def __init__(
        self,
        samples_file,
        embeddings_file,
        fasta_file,
        sample_genotype_file=None,
        genotype_file=None,
        flip_alleles=True,
        reverse_complement=True,
        jitter=0,
        noise=0,
        seed=None,
    ):
        super(VariantEmbedDataset, self).__init__(
            samples_file,
            embeddings_file,
            fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            seed=seed,
        )

        self.flip_alleles = flip_alleles

        self.genotype_file = genotype_file
        self.genotype_extr = None

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
                "logit_es",
            ]
        ).issubset(self.samples.keys())

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

    def get_phased_sequences(self, interval, sample_id, ref_pos):
        """
        Retrieve phased haplotype sequences for a given interval and sample, injecting
        phased and unphased variants as appropriate.

        Parameters
        ----------
        interval : GenomicInterval
            Genomic interval to extract.
        sample_id : str
            Sample identifier.
        ref_pos : int
            Position of the reference variant.

        Returns
        -------
        int
            Number of variants injected.
        str
            Haplotype 1 DNA sequence.
        str
            Haplotype 2 DNA sequence.
        """
        dna_seq_hap1 = self.fasta_extr[interval]
        dna_seq_hap2 = dna_seq_hap1.copy()

        # Get individual ID
        indiv_id = self.sample_to_genotype_df.loc[sample_id].indiv_id
        # Extract variants pertaining to individual
        variants = self.genotype_extr[interval]
        # TODO: Change this to df.query vs. str.contains?
        variants = variants[variants.indiv_id.str.contains(indiv_id)].set_index("start")

        assert len(variants) >= 1, "At least 1 variant is expected!"

        logger.debug(
            f"Found {len(variants)} variants in sample {sample_id} from individual {indiv_id}"
        )

        # Reference variant
        ref_phase_block = variants.loc[ref_pos].phase_block
        ref_gt = variants.loc[ref_pos].gt

        for v in variants.itertuples():
            # Variant position is the dataframe index
            pos = v.Index
            rel_pos = pos - interval.start

            # If variant is in the same phase block as ref variant, add it to
            # correct haplotype. This also adds the phased reference variant.
            if v.phase_block == ref_phase_block and ref_phase_block != ".":
                if v.gt == "1|0":
                    dna_seq_hap1 = (
                        dna_seq_hap1[:rel_pos] + v.alt + dna_seq_hap1[rel_pos + 1 :]
                    )
                elif v.gt == "0|1":
                    dna_seq_hap2 = (
                        dna_seq_hap2[:rel_pos] + v.alt + dna_seq_hap2[rel_pos + 1 :]
                    )
            # If the variant is the reference variant it is not phased, add
            # it to hap2 sequence
            elif pos == ref_pos and ref_phase_block == ".":
                dna_seq_hap2 = (
                    dna_seq_hap2[:rel_pos] + v.alt + dna_seq_hap2[rel_pos + 1 :]
                )
            # Add back the rest of the variants, they are unphased so we have
            # no idea which haplotype they are on. In this case, for all het
            # variants we just add the degenerate IUPAC character to both
            # haplotypes.
            else:
                if v.gt == "0/1" or v.gt == "1/0":
                    base = get_iupac_char_from_alleles((v.ref, v.alt))
                elif v.gt == "1/1":
                    base = v.alt
                else:
                    base = v.ref  # if reference do nothing
                dna_seq_hap1 = (
                    dna_seq_hap1[:rel_pos] + base + dna_seq_hap1[rel_pos + 1 :]
                )
                dna_seq_hap2 = (
                    dna_seq_hap2[:rel_pos] + base + dna_seq_hap2[rel_pos + 1 :]
                )

        # The hap1 sequence should have the reference allele for variant
        if ref_gt == "1|0":
            dna_seq_hap1, dna_seq_hap2 = dna_seq_hap2, dna_seq_hap1

        return (len(variants), dna_seq_hap1, dna_seq_hap2)

    def __getitem__(self, i):
        """
        Retrieve a single variant sample, including reference and alternate sequences,
        embedding, and variant metadata.

        Handles region jittering, reverse complementation, allele flipping, genotype injection,
        and embedding noise.

        Parameters
        ----------
        i : int
            Index of the variant to retrieve.

        Returns
        -------
        dict
            Dictionary with keys:
                - 'seq_ref': np.ndarray, one-hot encoded reference allele sequence
                - 'seq_alt': np.ndarray, one-hot encoded alternate allele sequence
                - 'embed': np.ndarray, cell-type/state embedding
                - 'ref_counts': float, reference allele counts
                - 'total_counts': float, total counts (reference + alternate)
                - 'bad_score': float, BAD score for the variant
                - 'lfc': float, log fold change (in natural log units)
                - 'sample_id': str, sample identifier
                - 'weight': float, sample weight (default 1.0)
        
        Notes
        -----
        The terminology "ref" vs. "alt" is a bit of a misnomer, as it is really
        haplotype 1 vs. haplotype 2. We call it "ref" vs. "alt" because the
        variant effect is always measured against the reference genome allele.
        """
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

        # TODO: move this to "get_phased_sequences" function?
        if self.include_genotypes and not self.genotype_extr:
            self.genotype_extr = TabixExtractor(self.genotype_file)

        chrom, pos, ref, alt, ref_counts, total_counts, bad, lfc, sample_id = (
            self.samples["chrom"][i].astype(str),
            self.samples["pos"][i],
            self.samples["ref"][i].astype(str),
            self.samples["alt"][i].astype(str),
            self.samples["ref_counts"][i].astype(np.float32),
            self.samples["total_counts"][i].astype(np.float32),
            self.samples["BAD"][i].astype(np.float32),
            self.samples["logit_es"][i].astype(np.float32),
            self.samples["sample_id"][i].astype(str),
        )

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen // 2)

        if self.jitter > 0:
            shift = self.random_state.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            _, dna_seq_hap1, dna_seq_hap2 = self.get_phased_sequences(
                interval, sample_id, pos
            )
        else:
            dna_seq_hap1 = self.fasta_extr[interval]
            dna_seq_hap2 = dna_seq_hap1[:pos] + alt + dna_seq_hap1[pos + 1 :]

        try:
            X_hap1, X_hap2 = (
                one_hot_encode(seq, dtype=np.float32)
                for seq in [dna_seq_hap1, dna_seq_hap2]
            )
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{variant.start} -- {sample_id})"
            )
            raise e

        # Random reverse complementation
        if self.reverse_complement and self.random_state.choice(2) == 1:
            X_hap1 = np.flip(X_hap1, [0, 1])
            X_hap2 = np.flip(X_hap2, [0, 1])

        # Flip reference and alternative alleles in input
        # for additional regularization
        if self.flip_alleles and self.random_state.choice(2) == 1:
            X_hap1, X_hap2 = X_hap2, X_hap1
            ref_counts = total_counts - ref_counts
            lfc = -1 * lfc

        # Cell type embeddings
        X_embed = self.embeddings_df[sample_id].to_numpy(dtype=np.float32)

        if self.noise > 0:
            X_embed = X_embed + self.random_state.normal(
                0, self.noise, len(X_embed)
            ).astype(np.float32)

        # The terminology "ref" vs. "alt" is a bit of a misnomer, as it is really
        # haplotype 1 vs. haplotype 2. We call it "ref" vs. "alt" because the
        # variant effect is always measured against the reference genome allele.
        return {
            "seq_ref": X_hap1.copy(),
            "seq_alt": X_hap2.copy(),
            "embed": X_embed.copy(),
            "ref_counts": ref_counts,
            "total_counts": total_counts,
            "bad_score": bad,
            "lfc": lfc * np.log(2),
            "sample_id": sample_id,
            "weight": 1.0,
        }
    
    def __del__(self):
        """
        Clean up open file handle for genotype extractor.
        """
        super(VariantEmbedDataset, self).__del__()

        if self.genotype_extr:
            self.genotype_extr.close()

    def __len__(self):
        """
        Return the number of variants in the dataset.

        Returns
        -------
        int
            Number of variants.
        """
        return self.samples["chrom"].shape[0]
