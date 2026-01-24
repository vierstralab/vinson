import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import gzip

from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

from vinson.utils.data_formatting import VinsonData
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
    data : VinsonData containing dict of sample metadata, encodings, embeddings
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
    """

    def __init__(
        self,
        data: VinsonData,
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
        self.genotype_file = genotype_file
        
        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.fasta_extr: FastaExtractor = None
        self.genotype_extr: TabixExtractor = None
        if self.genotype_file is not None:
            assert 'indiv_id' in self.data.keys(), "Sample to genotype mapping must include 'indiv_id' column."

            self.include_genotypes = True
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )
            self.include_genotypes = False

    def __del__(self):
        """
        Clean up open file handles for extractors.
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
        return len(self.data)

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
        x = self.data.embeddings_df.loc[sample_id].to_numpy(dtype=np.float32)

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
            indiv_id: str,
            reference_variant: VariantInterval=None
        ):
        """
        
        Returns:
            tuple: (base_sequence str, variants List[Variant])
        """
        seq = self.fasta_extr[interval]
        seq_iupac = seq_ref = seq_alt = str(seq) # modify all 3 regardless

        try:
            variants = self.genotype_extr[interval]
            if pd.isna(indiv_id) or indiv_id in ("None", ""):
                raise ValueError
        except ValueError:
            return 0, seq_iupac, seq_ref, seq_alt
        assert 'INDIV' in indiv_id, f"INDIV_ID format incorrect ({indiv_id}, {type(indiv_id)})."
        
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


class SequenceOnlyDataset(BaseSequenceDataset):
    """
    PyTorch Dataset for extracting sequence only (without cell-type embeddings), with optional
    genotype injection, negative sampling, and read depth normalization.
    
    Parameters
    ----------
    data : VinsonData 
        VinsonData object containing dict of sample metadata, encodings, embeddings
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
        data: VinsonData,
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
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
        )

        self.clip_density = clip_density
        self.min_bg = min_bg
        self.negatives_weight = negatives_weight

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
        data_slice = self.data[i]
        chrom = data_slice['chrom']
        summit = data_slice['summit']
        sample_id = data_slice['sample_id']
        density: np.ndarray = data_slice['density']
        bg = data_slice['background']
        read_depth = data_slice['read_depth']
        example_class = data_slice['class']

        is_multitask = density.ndim > 0

        # Define region
        interval = GenomicInterval(chrom, summit, summit).widen(self.seqlen // 2)

        # Jitter/shift region as necesary
        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        # indiv_id is expected to be in self.data if genotypes are included
        if self.include_genotypes:
            if is_multitask:
                raise ValueError("Multitask model does not support variant injection.")
            indiv_id = data_slice['indiv_id']
            
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
            # Reverse complete (augmentation)
            if self.reverse_complement and np.random.choice(2) == 1:
                ohe_seq = np.flip(ohe_seq, [0, 1])
        except ValueError as e:
            logger.error(
                f"Error converting DNA to one-hot encoding ({chrom}:{summit} -- {sample_id})"
            )
            raise e

        example_class = np.asarray(example_class != -1) # compatible with multitask
       
        # Adjust values as needed
        density = np.clip(density, None, self.clip_density)

        if 'dhs_weight' in data_slice:
            weight = data_slice['dhs_weight']
        else:
            weight = np.float32(1.0)

        weight_mult = np.where(example_class, 1.0, self.negatives_weight)

        # weight_mult = 1.0 if is_pos else self.negatives_weight

        weight = weight * weight_mult

        bg = np.clip(bg, self.min_bg, None)

        if not is_multitask:
            weight = weight.squeeze()
            example_class = example_class.squeeze()

        return {
            "ohe_seq": ohe_seq.copy(),
            "class": example_class,
            "density": density,
            "bg": bg,
            "read_depth": read_depth,
            "weight": weight,
            "chrom": chrom,
            "summit": summit,
            "sample_id": sample_id,
        }


class SequenceEmbedDataset(SequenceOnlyDataset):
    """
    PyTorch Dataset for extracting sequence and cell-type embeddings, with optional
    genotype injection, negative sampling, and read depth normalization.
    
    Parameters
    ----------
    data : VinsonData 
        VinsonData object containing dict of sample metadata, encodings, embeddings
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
        data = super().__getitem__(i)
        if data["sample_id"].ndim > 0:
            raise ValueError("Multitask model not supported in SequenceEmbedDataset.")
        
        # Get embeddings
        data['embed'] = self.get_embedding_vec(data['sample_id'])
        

        return data
