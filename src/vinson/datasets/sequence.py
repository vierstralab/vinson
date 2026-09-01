import numpy as np
import pandas as pd
import os 

from torch.utils.data import Dataset
import gzip

from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

from vinson.utils.data_formatting import VinsonData
from vinson.utils.sequence_utils import one_hot_encode, get_iupac_char_from_alleles
from vinson.utils.helpers import replace_at

from vinson.datasets.utils import TabixConnector, DataFrameConnector, IUPACSource
from vinson.utils.data_formatting.readers import parse_genotype_file
import logging

logger = logging.getLogger(__name__)
import warnings


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

        self.source = self.get_source()

        assert seqlen % 2 == 0, "Error 'seqlen' must be a even number!"
        self.seqlen = seqlen

        self.fasta_extr: FastaExtractor = None

        self.include_genotypes = False
        if genotype_file is not None:
            if isinstance(genotype_file, str):
                if genotype_file != '':
                    self.include_genotypes = True
            else:
                self.include_genotypes = True

        
        if self.include_genotypes:
            assert "indiv_id" in self.data.keys(), (
                "Sample to genotype mapping must include 'indiv_id' column."
            )
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )

    def get_connector(self):
        if isinstance(self.genotype_file, pd.DataFrame):
            return DataFrameConnector(self.genotype_file)
        else:
            return TabixConnector(self.genotype_file)

    def get_source(self):
        return IUPACSource(self.get_connector())

    def __del__(self):
        """
        Clean up open file handles for extractors.
        """
        if self.fasta_extr:
            self.fasta_extr.close()
    
    def _get_window(self, chrom, summit):
        interval = GenomicInterval(chrom, summit, summit).widen(self.seqlen // 2)
        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)
        return interval

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
        if not self.fasta_extr:
            self.fasta_extr = FastaExtractor(self.fasta_file)

    @staticmethod
    def _normalize_indiv_id(indiv_id):     
        if isinstance(indiv_id, (np.ndarray, list)):
            indiv_id = np.asarray(indiv_id).item()
        if pd.isna(indiv_id) or indiv_id in ("None", ""):
            indiv_id = ""
        else:
            assert isinstance(indiv_id, str), f"indiv_id must be str, got {type(indiv_id)}: {indiv_id}"
            assert (
                "INDIV" in indiv_id
            ), f"INDIV_ID format incorrect ({indiv_id}, {type(indiv_id)})."
        return indiv_id

        
    def get_sample_sequence(
            self,
            interval: GenomicInterval,
            indiv_id: str,
            reference_variant: VariantInterval=None,
        ):
        """
        Returns:
            tuple: (base_sequence str, variants List[Variant])
        """
        seq = self.fasta_extr[interval]

        indiv_id = self._normalize_indiv_id(indiv_id)

        if indiv_id == "":
            return str(seq)
    
        iupac_edits = self.source.get_edits(interval, indiv_id)
        seq_iupac = self.source.apply_edits(
            seq,
            interval_start=interval.start,
            edits=iupac_edits
        )
        return seq_iupac


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
        interval = self._get_window(chrom, summit)

        # indiv_id is expected to be in self.data if genotypes are included
        if self.include_genotypes:
            if is_multitask:
                raise ValueError("Multitask model does not support variant injection.")
            indiv_id = data_slice['indiv_id']
        else:
            indiv_id = ""

        dna_seq = self.get_sample_sequence(
            interval,
            indiv_id
        )

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
        if data["density"].ndim > 0:
            raise ValueError("Multitask model is not supported in SequenceEmbedDataset.")
        
        # Get embeddings
        data['embed'] = self.get_embedding_vec(data['sample_id'])
        return data
