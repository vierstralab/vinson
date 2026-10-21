import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from genome_tools import GenomicInterval, VariantInterval
from genome_tools.data.extractors import FastaExtractor

from vinson.utils.data_formatting import VinsonData

from vinson.datasets.utils import TabixConnector, DataFrameConnector, IUPACSource
from vinson.utils.sequence_utils import one_hot_encode
import logging

logger = logging.getLogger(__name__)


# TODO: move indiv parsing logic outside of the dataset
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

        if data.is_inference and (jitter or reverse_complement or noise):
            logger.warning(
                f"{type(self).__name__}: random augmentation enabled in an inference dataset "
                f"(jitter={jitter}, reverse_complement={reverse_complement}, noise={noise}); "
                "predictions will be non-deterministic."
            )

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

        self.source = self.get_source()
        if self.include_genotypes:
            assert "indiv_id" in self.data.keys(), (
                "include_genotypes = True. data should include 'indiv_id' field."
            )
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes."
            )

    def get_connector(self):
        if not self.include_genotypes:
            return None
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
    
    def _get_window(self, chrom, summit, shift: int = None):
        interval = GenomicInterval(chrom, summit, summit + 1).resize(self.seqlen)

        if shift is None:
            shift = np.random.randint(-self.jitter, self.jitter + 1) if self.jitter > 0 else 0

        interval.shift(shift, inplace=True)
        return interval, shift

    def _get_ohe_sequence(self, dna_seq, revcomp=None):
        ohe_seq = one_hot_encode(dna_seq.upper(), dtype=np.float32)
        # Reverse complement
        if revcomp is not None: # deterministic if present
            pass
        elif self.reverse_complement:
            revcomp = np.random.choice(2) == 1
        else:
            revcomp = False

        if revcomp:
            ohe_seq = np.flip(ohe_seq, [0, 1])

        return ohe_seq, revcomp

    def _slice_data(self, i: int):
        return self.data[i]

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

    def get_embedding_vec(self, sample_id, noise: float=None) -> np.ndarray:
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
        if self.data.embeddings_df is None:
            raise ValueError("data.embeddings_df is None but an embedding was requested")
        if self._is_multitask(sample_id):
            raise ValueError("Multitask model can't use embeddings.")

        x = self.data.embeddings_df.loc[sample_id].to_numpy(dtype=np.float32)

        # Add a little Gaussian noise to embeddings
        if (self.noise > 0) and (noise is None):
            noise = np.random.normal(0, self.noise, len(x)).astype(np.float32)
        elif noise is None:
            noise = 0

        x = x + noise
        return x, noise


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


    @staticmethod
    def _is_multitask(sample_id) -> bool:
        return not isinstance(sample_id, str)

