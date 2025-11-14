import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import gzip

from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

from vinson.utils.sequence_utils import one_hot_encode, get_iupac_char_from_alleles
from vinson.utils.helpers import replace_at
import logging

logger = logging.getLogger(__name__)



class BaseSequenceDataset(Dataset):
    """
    Base PyTorch Dataset for genomic sequence and embedding data.

    Provides common initialization and resource management for derived datasets,
    including loading sample metadata, embeddings, and setting up random state.

    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata. Must include keys like 'chrom'.
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    fasta_file : str
        Path to reference genome FASTA file.
    genotype_file : str, optional
        Path to genotype file in tabix format.
    reverse_complement : bool, default False
        Randomly reverse-complement sequences for augmentation.
    jitter : int, default 0
        Maximum number of bases to shift the region randomly.
    noise : float, default 0
        Standard deviation of Gaussian noise added to embeddings.
    seqlen : int, default 1344
        Sequence window length; must be even.

    Attributes
    ----------
    fasta_extr : FastaExtractor
        Reference genome sequence extractor.
    genotype_extr : TabixExtractor
        Genotype data extractor, if provided.
    seqlen : int
        Length of the sequence window.
    """

    def __init__(
        self,
        data: dict,
        embeddings_df: pd.DataFrame,
        fasta_file: str,
        genotype_file: str = None,
        reverse_complement=False,
        jitter=0,
        noise=0,
        seqlen=1344,
    ):
        self.fasta_file = fasta_file
        self.data = data
        self.reverse_complement = reverse_complement
        self.jitter = jitter
        self.noise = noise
        self.embeddings_df = embeddings_df
        self.genotype_file = genotype_file
        
        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.fasta_extr: FastaExtractor = None
        self.genotype_extr: TabixExtractor = None
        if self.genotype_file is not None:
            assert 'indiv_id' in data.keys(), "Sample to genotype mapping must include 'indiv_id' column."

            self.include_genotypes = True
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )
            self.include_genotypes = False


    def __del__(self):
        """
        Clean up open file handles for FASTA extractor.
        """
        if self.fasta_extr:
            self.fasta_extr.close()
        if self.genotype_extr:
            self.genotype_extr.close()

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
    
                    
    def _init_fileread(self):
        # pysam is not thread-safe
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)
        if self.include_genotypes and not self.genotype_extr:
            # Check header
            with gzip.open(self.genotype_file, "rt") as f:
                phased = "phase_set" in f.readline()
            if phased:
                print(f"[INFO] Detected phased genotype format ({self.genotype_file})")
                self.genotype_extr = TabixExtractor(
                    self.genotype_file,
                    skiprows=1,
                    columns=[
                        "chrom",
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
            else:
                print(f"[INFO] Using unphased genotype format ({self.genotype_file})")
                self.genotype_extr = TabixExtractor(
                    self.genotype_file,
                    columns=[
                        "chrom",
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
    
    def get_sample_sequence(
            self,
            interval: GenomicInterval,
            indiv_id,
            reference_variant: VariantInterval=None
        ):
        """
        
        Returns:
            tuple: (base_sequence str, variants List[Variant])
        """
        seq = self.fasta_extr[interval]
        seq_iupac = seq_ref = seq_alt = str(seq) # modify all 3 regardless

        if pd.isna(indiv_id) or indiv_id == "None":
            return 0, seq_iupac, seq_ref, seq_alt

        assert 'INDIV' in indiv_id, f"INDIV_ID format incorrect ({indiv_id})."
        variants = self.genotype_extr[interval]
        variants = variants[variants["indiv_id"] == f"{indiv_id}.bed.gz"]

        extra_columns = ('gt',)
        if reference_variant is not None:
            assert 'phase_set' in variants.columns, "Phased genotype data required for variant-aligned sequence extraction."
            try:
                # Look for the reference variant in the individual's genotypes
                phase_set = variants.set_index(
                    ["chrom", "start", "ref", "alt"]
                ).loc[
                    (
                        reference_variant.chrom,
                        reference_variant.start,
                        reference_variant.ref,
                        reference_variant.alt
                    ),
                    'phase_set'
                ]
                if not pd.isna(phase_set):
                    reference_variant.phase_set = phase_set
                    extra_columns = ('gt', 'phase_set') # extract phase set to match the reference variant
            except KeyError:
                raise ValueError(
                    "Query variant not found in genotyping file "
                    f"({str(interval)}/{indiv_id}/{reference_variant.pos}/{reference_variant.alt})"
                )
        variants = df_to_variant_intervals(
            variants, extra_columns=extra_columns
        )
        
        for variant_interval in variants:
            rel_pos = variant_interval.start - interval.start
            if 'phase_set' in extra_columns and reference_variant.phase_set == variant_interval.phase_set:
                    base = get_iupac_char_from_alleles(variant_interval.ref, variant_interval.alt)
                    seq_iupac = replace_at(seq_iupac, rel_pos, base)
                    if variant_interval.gt == "1|0":
                        seq_ref = replace_at(seq_ref, rel_pos, variant_interval.alt)
                        seq_alt = replace_at(seq_alt, rel_pos, variant_interval.ref)
                    elif variant_interval.gt == "0|1":
                        seq_ref = replace_at(seq_ref, rel_pos, variant_interval.ref)
                        seq_alt = replace_at(seq_alt, rel_pos, variant_interval.alt)
                    else:
                        raise ValueError(f'Phased genotype not recognized! {variant_interval}')
            else:
                assert variant_interval.gt[0] in ("0", "1") and variant_interval.gt[2] in ("0", "1"), f"Genotype format not recognized! {variant_interval} {variant_interval.gt}"
                variant_is_het = (
                    variant_interval.gt[0] == "1" and variant_interval.gt[2] == "0"
                ) or (
                    variant_interval.gt[0] == "0" and variant_interval.gt[2] == "1"
                )
                if variant_is_het:
                    base = get_iupac_char_from_alleles(variant_interval.ref, variant_interval.alt)
                    seq_iupac = replace_at(seq_iupac, rel_pos, base)
                    seq_ref = replace_at(seq_ref, rel_pos, variant_interval.ref)
                    seq_alt = replace_at(seq_alt, rel_pos, variant_interval.alt)
                elif variant_interval.gt[0] == "1":
                    base = variant_interval.alt
                    seq_iupac = replace_at(seq_iupac, rel_pos, base)
                    seq_ref = replace_at(seq_ref, rel_pos, base)
                    seq_alt = replace_at(seq_alt, rel_pos, base)
                else:
                    base = variant_interval.ref
                    seq_iupac = replace_at(seq_iupac, rel_pos, base)
                    seq_ref = replace_at(seq_ref, rel_pos, base)
                    seq_alt = replace_at(seq_alt, rel_pos, base)

        if reference_variant is not None:
            if reference_variant.gt == "1|0":
                seq_ref, seq_alt = seq_alt, seq_ref
        
            rel_pos = reference_variant.start - interval.start
            if (seq_ref[rel_pos] != reference_variant.ref) or (seq_alt[rel_pos] != reference_variant.alt):
                raise ValueError("Expected ref & alt alleles not found in correct position in sequences!", reference_variant, variants)

        return len(variants), seq_iupac, seq_ref, seq_alt


class SequenceEmbedDataset(BaseSequenceDataset):
    """
    PyTorch Dataset for extracting sequence and cell-type embeddings, with optional
    genotype injection, negative sampling, and read depth normalization.
    
    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata. Must include:
        'chrom', 'summit', 'class', 'density', 'sample_id', 'background', 'read_depth'.
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    fasta_file : str
        Path to reference genome FASTA file.
    genotype_file : str, optional
        Path to genotype file in tabix format. If provided, requires 'indiv_id' in data.
    negatives_weight : float, default 1.0
        Weight applied to negative class examples.
    clip_density : float, default 20
        Maximum value to clip density.
    min_bg : float, default 0.1
        Minimum value to clip background signal.
    reverse_complement : bool, default False
        Randomly reverse-complement sequences for augmentation.
    jitter : int, default 0
        Maximum number of bases to shift sequences.
    noise : float, default 0
        Standard deviation of Gaussian noise added to embeddings.

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
            genotype_file=genotype_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
        )

        self.clip_density = clip_density
        self.min_bg = min_bg
        self.negatives_weight = negatives_weight
        
        #moved from base dataset
        assert set(
            [
                "chrom", "summit", "class", 
                "density", "sample_id", 
                "background", "read_depth"
            ]
        ).issubset(
            self.data.keys()
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
            - 'ohe_seq': one-hot encoded DNA sequence (np.ndarray)
            - 'embed': cell-type embedding (np.ndarray)
            - 'class': int, 1 (positive) or -1 (negative)
            - 'density': float, clipped density value
            - 'bg': float, clipped background
            - 'read_depth': float, read depth
            - 'weight': float, sample weight
            - 'chrom': str, chromosome
            - 'summit': int, center coordinate
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
            _, dna_seq, _, _ = self.get_sample_sequence(
                interval,
                indiv_id
            )
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


class VariantEmbedDataset(BaseSequenceDataset):
    """
    PyTorch Dataset for variant effect prediction with reference and alternate sequences.

    Parameters
    ----------
    data : dict
        Dictionary containing variant metadata. Must include:
        'chrom', 'pos', 'ref', 'alt', 'ref_counts', 'total_counts', 'BAD', 'sample_id', 'logit_es'.
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    fasta_file : str
        Path to reference genome FASTA file.
    genotype_file : str
        Path to genotype file in tabix format. Requires 'indiv_id' in data.
    flip_alleles : bool, default True
        Randomly swap reference and alternate sequences for augmentation.
    reverse_complement : bool, default True
        Randomly reverse-complement sequences for augmentation.
    jitter : int, default 0
        Maximum number of bases to shift sequences randomly.
    noise : float, default 0
        Standard deviation of Gaussian noise added to embeddings.
    """

    def __init__(
        self,
        data: dict,
        embeddings_df: pd.DataFrame,
        fasta_file: str,
        genotype_file: str = None,
        flip_alleles=True,
        reverse_complement=True,
        jitter=0,
        noise=0,
    ):
        super().__init__(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
        )

        self.flip_alleles = flip_alleles
        self.genotype_extr: TabixExtractor = None

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
        ).issubset(self.data.keys())

        if self.genotype_file is not None:
            assert 'indiv_id' in data.keys(), "Sample to genotype mapping must include 'indiv_id' column."

            self.include_genotypes = True
        else:
            raise ValueError(
                "Genotype file is required but not provided. "
                "Please specify `genotype_file` to enable variant-aware training."
            )
  

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
        self._init_fileread()

        chrom, pos, ref, alt, ref_counts, total_counts, bad, lfc, sample_id = (
            self.data["chrom"][i].astype(str),
            self.data["pos"][i],
            self.data["ref"][i].astype(str),
            self.data["alt"][i].astype(str),
            self.data["ref_counts"][i].astype(np.float32),
            self.data["total_counts"][i].astype(np.float32),
            self.data["BAD"][i].astype(np.float32),
            self.data["logit_es"][i].astype(np.float32),
            self.data["sample_id"][i].astype(str),
        )

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen // 2)

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        rel_pos = pos - interval.start

        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            indiv_id = self.data["indiv_id"][i]   
            _, _, dna_seq_ref, dna_seq_alt = self.get_sample_sequence(
                interval, 
                indiv_id,
                reference=VariantInterval(
                    chrom=chrom, start=pos, end=pos + 1, ref=ref, alt=alt
                )
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

    def __len__(self):
        """
        Return the number of variants in the dataset.

        Returns
        -------
        int
            Number of variants.
        """
        return self.data["chrom"].shape[0]
