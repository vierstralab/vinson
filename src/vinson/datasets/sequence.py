import numpy as np

from genome_tools import GenomicInterval


from vinson.utils.data_formatting import VinsonData
from vinson.utils.sequence_utils import one_hot_encode

import logging

from .base import BaseSequenceDataset


logger = logging.getLogger(__name__)


class InferenceSequenceOnlyDataset(BaseSequenceDataset):

    def _slice_data(self, i: int):
        return self.data[i]

    def _parse_position(self, data_dict: dict) -> GenomicInterval:
        chrom = data_dict['chrom']
        summit = data_dict['summit']
        shift = int(data_dict.get('shift', 0))

        interval, shift = self._get_window(chrom, summit, shift=shift)
        return interval, shift

    def _parse_sequence(self, data_dict: dict, interval: GenomicInterval):
        self._init_fileread()
        sample_id = data_dict['sample_id']

        if self.include_genotypes:
            if self._is_multitask(sample_id):
                raise ValueError("Multitask model does not support variant injection.")
            indiv_id = data_dict['indiv_id']
        else:
            indiv_id = ""

        dna_seq = self.get_sample_sequence(interval, indiv_id)

        if 'revcomp' in data_dict:                       
            revcomp = data_dict['revcomp']
        else:
            revcomp = None
        ohe_seq, revcomp = self._get_ohe_sequence(dna_seq=dna_seq, revcomp=True)

        return ohe_seq.copy(), revcomp

    def get_sample_sequence(
            self,
            interval: GenomicInterval,
            indiv_id: str,
        ):
        """
        Returns:
            tuple: genomic sequence with variants encoded as IUPAC symbols
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
            - 'chrom': str, chromosome
            - 'summit': int, center coordinate
            - 'sample_id': str, sample identifier
        
            Optional, for inference:
            - 'revcomp': bool, whether take revcomp of the sequence
        """
        data_slice = self._slice_data(i)

        interval, shift = self._parse_position(data_slice)
        ohe_seq, revcomp = self._parse_sequence(data_slice, interval)

        return {
            "ohe_seq": ohe_seq.copy(),
            "chrom": data_slice['chrom'],
            "summit": data_slice['summit'],
            "sample_id": data_slice['sample_id'],
            "shift": shift,
            "ohe_revcomp": revcomp
        }


class InferenceSequenceEmbedDataset(InferenceSequenceOnlyDataset):
    def __init__(self, data, fasta_file, genotype_file = None, seqlen=1344): # added because we don't need random noise and revcomp as augmentation
            super().__init__(data, fasta_file, genotype_file, seqlen=seqlen)

    def __getitem__(self, i):
        self._init_fileread()
        data = self.data[i]
        interval, shift = self._parse_position(data)
        ohe_seq, revcomp = self._parse_sequence(data, interval)
        embed = self.get_embedding_vec(data['sample_id'])
        
        return {
                "ohe_seq": ohe_seq.copy(),
                "chrom": data['chrom'],
                "summit": data['summit'],
                "sample_id": data['sample_id'],
                "shift": shift,
                "revcomp": revcomp,
                "embed": embed,

            }

# class InferenceSequenceEmbedDataset(BaseSequenceDataset):
    

#     def _parse_position(self, data_dict: dict) -> GenomicInterval:
#             chrom = data_dict['chrom']
#             summit = data_dict['summit']
#             shift = int(data_dict.get('shift', 0))

#             interval, shift = self._get_window(chrom, summit, shift=shift)
#             return interval, shift
    
#     def get_sample_sequence(
#             self,
#             interval: GenomicInterval,
#             indiv_id: str,
#         ):
#         """
#         Returns:
#             tuple: genomic sequence with variants encoded as IUPAC symbols
#         """
#         seq = self.fasta_extr[interval]

#         indiv_id = self._normalize_indiv_id(indiv_id)

#         if indiv_id == "":
#             return str(seq)
    
#         iupac_edits = self.source.get_edits(interval, indiv_id)
#         seq_iupac = self.source.apply_edits(
#             seq,
#             interval_start=interval.start,
#             edits=iupac_edits
#         )
#         return seq_iupac

#     def _parse_sequence(self, data_dict: dict, interval: GenomicInterval): # it should give revcomp always, it's inference dataset
#         self._init_fileread()            
#         if self.include_genotypes:
#             indiv_id = data_dict['indiv_id']
#         else:
#             indiv_id = ""

#         dna_seq = self.get_sample_sequence(interval, indiv_id)

#         ohe_seq = one_hot_encode(dna_seq)
#         revcomp = np.flip(ohe_seq, [0,1])
#         return ohe_seq.copy(), revcomp.copy()
    
#     def __getitem__(self, i):
#         self._init_fileread()
#         data = self.data[i]
#         interval, shift = self._parse_position(data)
#         ohe_seq, revcomp = self._parse_sequence(data, interval)
#         embed, noise = self.get_embedding_vec(data['sample_id'])
        
#         return {
#                 "ohe_seq": ohe_seq.copy(),
#                 "chrom": data['chrom'],
#                 "summit": data['summit'],
#                 "sample_id": data['sample_id'],
#                 "shift": shift,
#                 "revcomp": revcomp,
#                 "embed": embed,
#                 "noise":noise

#             }

class SequenceOnlyDataset(InferenceSequenceOnlyDataset):
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
        seqlen=1344
    ):
        super().__init__(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            seqlen=seqlen
        )

        self.clip_density = clip_density
        self.min_bg = min_bg
        self.negatives_weight = negatives_weight

        assert not data.is_inference, f'Sequency only dataset requires targets to be present in VinsonData data.is_inference={data.is_inference}. Use infererence dataset instead.'


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
        data_slice = self._slice_data(i)

        interval, shift = self._parse_position(data_slice)

        ohe_seq, revcomp = self._parse_sequence(data_slice, interval)

        result = {
            "ohe_seq": ohe_seq.copy(),
            "chrom": data_slice['chrom'],
            "summit": data_slice['summit'],
            "sample_id": data_slice['sample_id'],
            "shift": shift,
            "revcomp": revcomp
        }

        # parse target data
        sample_id = data_slice.get('sample_id', None) 
        is_multitask = self._is_multitask(sample_id) # method requires sample_id

        density: np.ndarray = data_slice['density']
        bg = data_slice['background']
        read_depth = data_slice['read_depth']
        example_class = data_slice['class']
        example_class = np.asarray(example_class == 1) # compatible with multitask
       
        # Adjust values as needed
        density = np.clip(density, None, self.clip_density)

        if 'dhs_weight' in data_slice:
            weight = data_slice['dhs_weight']
        else:
            weight = np.float32(1.0)

        weight_mult = np.where(example_class, 1.0, self.negatives_weight)

        weight = weight * weight_mult

        if not is_multitask:
            weight = weight.squeeze()
            example_class = example_class.squeeze()

        bg = np.clip(bg, self.min_bg, None)

        result.update(
            {
                "class": example_class,
                "density": density,
                "bg": bg,
                "read_depth": read_depth,
                "weight": weight,
            }
        )
        return result


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
        data['embed'] = self.get_embedding_vec(data['sample_id'])
        return data
