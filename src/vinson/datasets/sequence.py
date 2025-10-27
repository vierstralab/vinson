import numpy as np
import pandas as pd
from collections import namedtuple

from torch.utils.data import Dataset

from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

from vinson.utils.sequence_utils import one_hot_encode, get_iupac_char_from_alleles

import logging

logger = logging.getLogger(__name__)


class BaseSequenceDataset(Dataset):
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
    seqlen : int
        Length of the sequence window.
    """

    def __init__(
        self,
        data: dict,
        embeddings_df: pd.DataFrame,
        fasta_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
        seqlen=1344,
    ):
        self.fasta_file = fasta_file
        self.data = data

        # TODO : validate data keys depending on model type
        assert set(
            [
                "chrom", "summit", "class", 
                "density", "sample_id", 
                "background", "read_depth"
            ]
        ).issubset(
            self.data.keys()
        )

        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise
        self.embeddings_df = embeddings_df

        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.fasta_extr: FastaExtractor = None

    def __del__(self):
        """
        Clean up open file handles for samples and FASTA extractor.
        """
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
        """
        return len(self.data['chrom'])

    def get_embedding_vec(self, sample_id) -> np.ndarray:
        """
        Get the embedding vector a sample id

        Parameters
        ----------
        sample_id : str
            Sample id

        Returns
        -------
        np.ndarray
            The embedding vector that sample.
        """
        # Cell type/state embeddings
        x = self.embeddings_df.loc[sample_id].to_numpy(dtype=np.float32)

        # Add a little Gaussian noise to embeddings
        if self.noise > 0:
            x = x + np.random.normal(0, self.noise, len(x)).astype(np.float32)

        return x


class SequenceEmbedDataset(BaseSequenceDataset):
    """
    PyTorch Dataset for extracting sequence and cell-type embeddings, with optional
    genotype injection, negative sampling, and read depth normalization.

    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata and values.
    embeddings_file : str
        Path to tab-delimited file with cell-type/state embeddings.
    fasta_file : str
        Path to reference genome FASTA file.
    genotype_file : str, optional
        Path to genotype file in tabix format.

    negative_samples_weight : float, optional
        Weight assigned to negative samples (default: 1).
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
    """

    def __init__(
        self,
        data: dict,
        embeddings_df: pd.DataFrame,
        fasta_file: str,
        genotype_file: str = None,
        negatives_weight: float = 1.0,
        clip_density=20,
        min_bg=0.1,
        reverse_complement=False,
        jitter=0,
        noise=0,
    ):
        super().__init__(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
        )

        self.clip_density = clip_density
        self.min_bg = min_bg
        self.negatives_weight = negatives_weight

        self.genotype_file = genotype_file
        self.genotype_extr: TabixExtractor = None
        if genotype_file is not None:
            assert 'indiv_id' in data.keys(), "Sample to genotype mapping must include 'indiv_id' column."

            self.include_genotypes = True
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )
            self.include_genotypes = False

    def get_sample_sequence(self, interval, sample_id, indiv_id=None):
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

        Notes
        -----
        Works with both unphased and phased genotypes.
        """
        seq = self.fasta_extr[interval]

        # Check if sample_id has a genotype
        if pd.isna(indiv_id):
            logging.debug(
                f"INDIV_ID for {sample_id} not provided. ({str(interval)})"
            )
            # Return the original sequence
            return 0, seq

        # Extract variants pertaining to individual
        variants = self.genotype_extr[interval]
        
        assert 'INDIV' in indiv_id, f"INDIV_ID format incorrect ({indiv_id})."

        # FIX formatting issue (.bed.gz suffix) 
        variants = variants[variants["indiv_id"] == f"{indiv_id}.bed.gz"].set_index(
            ["chr", "start", "ref", "alt"]
        )

        logger.debug(
            f"Found {len(variants)} variants in sample {sample_id} from individual {indiv_id}"
        )

        for v in variants.itertuples():
            # Variant position is the dataframe index
            _chr, _pos, _ref, _alt = v.Index
            rel_pos = _pos - interval.start

            # If heterozygous, get IUPAC base character
            # make parsing function more robust
            if (v.gt[0] == "1" and v.gt[2] == "0") or (
                v.gt[0] == "0" and v.gt[2] == "1"
            ):
                base = get_iupac_char_from_alleles((_ref, _alt))
            # If homozygous alternate
            elif v.gt[0] == "1" and v.gt[2] == "1":
                base = _alt
            # Else homozygous reference
            else:
                base = _ref

            seq = seq[:rel_pos] + base + seq[rel_pos + 1 :]

        return len(variants), seq


    def _init_fileread(self):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

        # tabix is not thread-safe
        if self.include_genotypes and not self.genotype_extr:
            self.genotype_extr = TabixExtractor(
                self.genotype_file,
                columns=[
                    "chr",
                    "start",
                    "end",
                    "rs_id",
                    "ref",
                    "alt",
                    "af_ref",
                    "af_alt",
                    "gt",
                    "_0",
                    "_1",
                    "_2",
                    "_3",
                    "indiv_id",
                ],
                na_values=".",
            )


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
        self._init_fileread()
        chrom, summit, sample_id, density, bg, read_depth, example_class = (
            self.data["chrom"][i],
            self.data["summit"][i],
            self.data["sample_id"][i],
            self.data["density"][i],
            self.data["background"][i],
            self.data["read_depth"][i],
            self.data["class"][i],
        )
        assert example_class in [-1, 1], "Class must be -1 or 1."

        # Define region
        interval = GenomicInterval(chrom, summit, summit).widen(self.seqlen // 2)

        # Jitter/shift region as necesary
        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        # indiv_id is expected to be in self.data if genotypes are included
        if self.include_genotypes:
            indiv_id = self.data['indiv_id'][i]
            _, dna_seq = self.get_sample_sequence(interval, sample_id, indiv_id)
        else:
            dna_seq = self.fasta_extr[interval]
        

        # One-hot encode DNA sequence
        #added upper for mouse fasta
        try:
            ohe_seq = one_hot_encode(dna_seq.upper(), dtype=np.float32)
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{summit} -- {sample_id})"
            )
            raise e

        # Reverse complete (augmentation)
        if self.reverse_complement and np.random.choice(2) == 1:
            ohe_seq = np.flip(ohe_seq, [0, 1])

        # Get embeddings
        embed = self.get_embedding_vec(sample_id)
       
        # Adjust values as needed
        density = np.clip(density, None, self.clip_density)

        if 'dhs_weight' in self.data:
            weight = self.data['dhs_weight'][i]
        else:
            weight = np.float32(1.0)

        weight_mult = 1.0 if example_class == 1 else self.negatives_weight
        weight = weight * weight_mult

        bg = np.clip(bg, self.min_bg, None)

        return {
            "ohe_seq": ohe_seq.copy(),
            "embed": embed.copy(),
            "class": example_class,
            "density": density,
            "bg": bg,
            "read_depth": read_depth,
            "weight": weight,
            "chrom": chrom,
            "summit": summit,
            "sample_id": sample_id,
        }


    def __del__(self):
        """
        Clean up open file handles for genotype and negative sample extractors.
        """
        super().__del__()

        if self.genotype_extr:
            self.genotype_extr.close()




Variant = namedtuple("Variant", ["chr", "pos", "ref", "alt"])

class VariantEmbedDataset(BaseSequenceDataset):
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
    ):
        super().__init__(
            samples_file,
            embeddings_file,
            fasta_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
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

    def get_phased_sequences(self, interval, sample_id, reference):
        """
        Retrieve phased haplotype sequences for a given interval and sample, injecting
        phased and unphased variants as appropriate.

        Parameters
        ----------
        interval : GenomicInterval
            Genomic interval to extract.
        sample_id : str
            Sample identifier.
        query_pos : int
            Position of the query variant.

        Returns
        -------
        int
            Number of variants injected.
        str
            Haplotype 1 DNA sequence.
        str
            Haplotype 2 DNA sequence.
        """
        seq_ref = self.fasta_extr[interval]
        seq_alt = str(seq_ref)

        # Check if sample in has a genotype
        if sample_id not in self.sample_to_genotype_df.index:
            raise ValueError(f"Sample {sample_id} not in samples to genotype file!")

        # Get individual ID from sample ID
        indiv_id = self.sample_to_genotype_df.loc[sample_id, "indiv_id"]

        # Check if not NaN
        if pd.isna(indiv_id):
            raise ValueError(
                f"No INDIV_ID for {sample_id} samples to genotype file (indiv_id = NaN). ({str(interval)})"
            )

        # Extract variants pertaining to individual
        variants = self.genotype_extr[interval]
        variants = variants[variants["indiv_id"].str.contains(indiv_id)].set_index(
            ["chr", "start", "ref", "alt"]
        )

        try:
            # Look for query variant
            ref_variant = variants.loc[reference]
        except KeyError:
            raise ValueError(
                "Query variant not found in genotyping file "
                f"({str(interval)}/{sample_id}/{indiv_id}/{reference.pos})"
            )

        logger.info(
            f"Found {len(variants)} variants in sample {sample_id} from individual {indiv_id} in {interval}"
        )

        for v in variants.itertuples():
            # Variant position is the dataframe index
            _chr, _pos, _ref, _alt = v.Index
            rel_pos = _pos - interval.start
            # Phased variants, including reference
            if v.phase_block == ref_variant.phase_block and not pd.isna(
                ref_variant.phase_block
            ):
                if v.gt == "1|0":
                    seq_ref = seq_ref[:rel_pos] + _alt + seq_ref[rel_pos + 1 :]
                elif v.gt == "0|1":
                    seq_alt = seq_alt[:rel_pos] + _alt + seq_alt[rel_pos + 1 :]
                else:
                    raise ValueError(
                        f"Phased genotype {v.gt} not recognized! ({_chr}:{_pos}:{indiv_id})"
                    )
            # The unphased reference
            elif v.Index == reference:
                seq_alt = seq_alt[:rel_pos] + _alt + seq_alt[rel_pos + 1 :]
            # Unphased additional variants that are not at the same 
            # position as the referencee
            elif _pos != reference.pos:
                # If heterozygous, get IUPAC base character
                if (v.gt[0] == "1" and v.gt[2] == "0") or (
                    v.gt[0] == "0" and v.gt[2] == "1"
                ):
                    base = get_iupac_char_from_alleles((_ref, _alt))
                # If homozygous alternate
                elif v.gt[0] == "1" and v.gt[2] == "1":
                    base = _alt
                # Else homozygous reference
                else:
                    base = _ref

                seq_ref = seq_ref[:rel_pos] + base + seq_ref[rel_pos + 1 :]
                seq_alt = seq_alt[:rel_pos] + base + seq_alt[rel_pos + 1 :]
            else:
                pass

        # The ref sequence should have the reference allele for
        # variant. This would only occur for phased variants, hence
        # the "1|0" genotype.
        if ref_variant["gt"] == "1|0":
            seq_ref, seq_alt = seq_alt, seq_ref

        # Check the sequences
        rel_pos = reference[1] - interval.start
        if (seq_ref[rel_pos] != reference[2]) or (seq_alt[rel_pos] != reference[3]):
            raise ValueError("Expected ref & alt alleles not found in correct position in sequences!", reference, variants)

        return (len(variants), seq_ref, seq_alt)

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
            self.genotype_extr = TabixExtractor(
                self.genotype_file,
                skiprows=1,
                columns=[
                    "chr",
                    "start",
                    "end",
                    "ref",
                    "alt",
                    "indiv_id",
                    "gt",
                    "phase_block",
                ],
                na_values={"phase_block": "."},
            )

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
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        rel_pos = pos - interval.start

        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            _, dna_seq_ref, dna_seq_alt = self.get_phased_sequences(
                interval, sample_id, Variant(chrom, pos, ref, alt)
            )
        else:
            dna_seq_ref = self.fasta_extr[interval]
            dna_seq_alt = dna_seq_ref[:rel_pos] + alt + dna_seq_ref[rel_pos + 1 :]

        try:
            ohe_seq_ref, ohe_seq_alt = (
                one_hot_encode(seq, dtype=np.float32)
                for seq in [dna_seq_ref, dna_seq_alt]
            )
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{variant.start} -- {sample_id})"
            )
            raise e

        # Random reverse complementation
        if self.reverse_complement and np.random.choice(2) == 1:
            ohe_seq_ref = np.flip(ohe_seq_ref, [0, 1])
            ohe_seq_alt = np.flip(ohe_seq_alt, [0, 1])

        # Flip reference and alternative alleles in input
        # for additional regularization
        if self.flip_alleles and np.random.choice(2) == 1:
            ohe_seq_ref, ohe_seq_alt = ohe_seq_alt, ohe_seq_ref
            ref_counts = total_counts - ref_counts
            lfc = -1 * lfc

        # Cell type embeddings
        embed = self.get_embedding_vec(sample_id)

        # The terminology "ref" vs. "alt" is a bit of a misnomer, as it is really
        # haplotype 1 vs. haplotype 2. We call it "ref" vs. "alt" because the
        # variant effect is always measured against the reference genome allele.
        return {
            "ohe_seq_ref": ohe_seq_ref.copy(),
            "ohe_seq_alt": ohe_seq_alt.copy(),
            "embed": embed.copy(),
            "ref_counts": np.float32(ref_counts),
            "total_counts": np.float32(total_counts),
            "bad_score": np.float32(bad),
            "lfc": lfc * np.log(2),
            "sample_id": sample_id,
            "weight": 1.0,
            "chrom": chrom,
            "pos": pos,
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
