from os.path import exists
import numpy as np
import pandas as pd

from genome_tools import GenomicInterval, VariantInterval
from vinson.utils.helpers import replace_at
from vinson.utils.data_formatting import VinsonData
from vinson.utils.sequence_utils import one_hot_encode

from .sequence import BaseSequenceDataset,CartesianInferenceDataset, logger

from genome_tools.data.extractors import FastaExtractor
import warnings

from vinson.utils.helpers import replace_at


class VariantEmbedDataset(BaseSequenceDataset):
    """
    PyTorch Dataset for variant effect prediction with reference and alternate sequences.

    Parameters
    ----------
    data : VinsonData
        VinsonData containing dict of variant metadata, categorical encodings and embeddings.
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
        data: VinsonData,
        fasta_file: str,
        genotype_file: str = None,
        flip_alleles=False,
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

        self.flip_alleles = flip_alleles
        self.fasta_extr: FastaExtractor = None
        self.genotype_file = genotype_file
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
            assert 'indiv_id' in self.data.keys(), "Sample to genotype mapping must include 'indiv_id' column."

            self.include_genotypes = True
        else:
            self.include_genotypes = False
  

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

        data_slice = self.data[i]
        chrom = data_slice['chrom']
        pos = data_slice['pos']
        ref = data_slice['ref']
        alt = data_slice['alt']
        ref_counts = data_slice['ref_counts']
        total_counts = data_slice['total_counts']
        bad = data_slice['BAD']
        lfc = data_slice['logit_es']
        sample_id = data_slice['sample_id']

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen // 2)

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        rel_pos = pos - interval.start
        
        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            indiv_id = data_slice['indiv_id']   
            _, _, dna_seq_ref, dna_seq_alt = self.get_sample_sequence(
                interval, 
                indiv_id,
                reference_variant=VariantInterval(
                    chrom=chrom, start=pos-1, end=pos, ref=ref, alt=alt,
                ),
            )
        else:
            dna_seq_ref = dna_seq_alt = self.fasta_extr[interval]
            start = pos-1 
            end = pos
            rel = start - interval.start
            dna_seq_ref = replace_at(dna_seq_ref, rel, ref)
            dna_seq_alt = replace_at(dna_seq_alt, rel, alt)
            # changed: end is pos and start is pos-1 need to be changed to match 
            # dna_seq_alt = dna_seq_ref[:rel_pos] + alt + dna_seq_ref[rel_pos + 1 :]
        if len(dna_seq_ref) != 1344 or len(dna_seq_alt) != 1344:
            warnings.warn(
            f"[DEBUG WARNING] idx={i}, chrom={chrom}, pos={pos}, "
            f"ref_seq length={len(dna_seq_ref)} alt_len expected seqlen={len(dna_seq_alt)}, relpos {rel_pos}, alt {alt}, ref {ref}"
        )

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


class VariantInferenceDataset(BaseSequenceDataset):

    def __init__(self, data: VinsonData, strict_ref_check=True, **kwargs):
        super().__init__(
            data=data,
            **kwargs
        )
        self.strict_ref_check = strict_ref_check
        self.include_genotypes = False

    def __getitem__(self, i):
        self._init_fileread()
        row = self.data[i]
        
        chrom = row["chrom"]
        start = row["pos"] - 1# 0-based
        ref = row["ref"]
        alt = row["alt"]
        sample_id = row["sample_id"]

        # base genomic context
        interval = self._get_window(chrom, start)
        seq = self.fasta_extr[interval].upper()

        # enforce alleles
        center = start - interval.start

        center_base = seq[center]
        # optional sanity check
        msg = f"Alleles mismatch at {chrom}:{start} for sample {sample_id}: expected {ref}/{alt}, got {seq[center]}"
        if self.strict_ref_check:
            assert center_base in (ref, alt), msg
        elif center_base not in (ref, alt):
            logger.warning(msg + ". Proceeding anyway (strict_ref_check=False).")

        seq_ref = replace_at(seq, center, ref)
        seq_alt = replace_at(seq, center, alt)

        ohe_ref = one_hot_encode(seq_ref)
        ohe_alt = one_hot_encode(seq_alt)

        embed = self.get_embedding_vec(sample_id)

        return {
            "ohe_seq_ref": ohe_ref,
            "ohe_seq_alt": ohe_alt,
            "center_seq": center_base,
            'seq': seq,
            "ref": ref,
            "alt": alt,
            "embed": embed,
            "chrom": chrom,
            "start": start,
        }


class CartesianVariantInferenceDataset(CartesianInferenceDataset):
    """
    A PyTorch Dataset that generates all combinations of genetic variants, 
    cell-type embeddings, and shifts for model inference.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        DataFrame containing variant coordinates and alleles. 
        Must include columns: ['chrom', 'pos', 'ref', 'alt'].
    embed_df : pandas.DataFrame
        DataFrame of cell-type or state embeddings. The index should represent 
        the sample/embedding ID, and the columns should contain the numeric embedding values.
    fasta_file : str
        Path to the reference genome FASTA file.
    shifts : ArrayLike, optional
        List of spatial shifts (in base pairs) to offset the sequence window. 
        Default is [0].
    seqlen : int, optional
        The fixed total length of the output DNA sequence. Default is 1344.
    """

    REQUIRED_COLUMNS = ['chrom', 'pos', 'ref', 'alt']
    
    def __init__(self, 
                 variants_df: pd.DataFrame, 
                 embed_df: pd.DataFrame,
                 fasta_file: str,
                 embed_meta_df: pd.DataFrame = None,
                 genotype_file: str = None,
                 shifts: list = [0], 
                 seqlen: int = 1344,
                 
                 ):
        assert_flag = all([col in variants_df.columns for col in self.REQUIRED_COLUMNS])
        assert assert_flag, 'Ensure all required columns are present'
        
        self.fasta_file = fasta_file
        self.fasta_extr = None
        self.genotype_extr = None 

        self.embeds_ids = embed_df.index.values
        self.embeds_vals = embed_df.astype(np.float32).values
    
        is_genotype = genotype_file is not None and exists(genotype_file)
        is_embed_meta = embed_meta_df is not None 
        self.include_genotypes = is_embed_meta and is_genotype
        
        if self.include_genotypes:
            assert "indiv_id" in embed_meta_df.columns, (
                "Sample to genotype mapping must include 'indiv_id' column.")

            self.embed_meta_df = embed_meta_df.loc[self.embeds_ids].copy()
            self.indiv_ids = self.embed_meta_df['indiv_id'].values
            self.genotype_file = genotype_file
        else:
            logger.info(
                "No genotyping files provided -- continuing without sample genotypes.")
        
        self.coords_chrom = variants_df['chrom'].values
        self.coords_pos = variants_df['pos'].values
        self.coords_ref = variants_df['ref'].values
        self.coords_alt = variants_df['alt'].values
        self.coords_df = variants_df.copy()
        
        self.shifts = np.asarray(shifts)
        self.seqlen = seqlen
        
        self.shape = (len(embed_df), len(self.shifts), len(variants_df))
    
    def _prepare_alleles(self, dna_seq, rel_pos, ref, alt):
        """
        Extracts the reference sequence and constructs the alternative sequence.

        Parameters
        ----------
        interval : GenomicInterval
            The target genomic window.
        pos : int
            The absolute genomic coordinate of the variant.
        ref : str
            The reference allele.
        alt : str
            The alternative allele.

        Returns
        -------
        tuple of str
            The reference and alternative DNA sequences (uppercase).
        """

        dna_seq_ref = dna_seq_alt = dna_seq #self.fasta_extr[interval]
        dna_seq_ref = replace_at(dna_seq_ref, rel_pos, ref)

        if alt in ('', '.', '-', '_'): # in case there is deletion
            alt = ''
            interval = interval.widen(left=0, right=1)
            dna_seq_alt = self.fasta_extr[interval]

        dna_seq_alt = self._crop(replace_at(dna_seq_alt, rel_pos, alt))
        return dna_seq_ref.upper(), dna_seq_alt.upper()
    
    def prepare_meta(self):
        """
        Generates a flat metadata DataFrame matching the exact iteration order of __getitem__.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing variant details, shift values, embedding IDs, 
            and a unique string identifier (`var_id`) for every sample in the dataset.
        """
        dtypes = {
            'ref' : 'category',
            'alt' : 'category',
            'chrom': 'category',
            'embed_id': 'category',      
            'shift': 'category',     
            'var_id': 'str'  
        }
        idxs = np.arange(len(self))
        embeds_idxs, shift_idxs, coord_idxs = np.unravel_index(idxs, self.shape)
        
        df = self.coords_df.iloc[coord_idxs].copy()
        df['shift'] = self.shifts[shift_idxs]
        df['embed_id'] = self.embeds_ids[embeds_idxs]
        df['var_id'] = df.chrom.astype(str) + '_' + df.pos.astype(str) + '_' + df.ref + '_' + df.alt
        
        # df = df.astype(dtypes)
        return df.reset_index(drop=True)

    def __getitem__(self, idx):
        """
        Retrieves a single batch item containing one-hot encoded alleles and metadata.

        Parameters
        ----------
        idx : int
            The flattened integer index of the dataset.

        Returns
        -------
        dict
            A dictionary containing:
            - 'ref': One-hot encoded reference sequence.
            - 'ref_revcomp': Reverse complement of the reference sequence.
            - 'alt': One-hot encoded alternative sequence.
            - 'alt_revcomp': Reverse complement of the alternative sequence.
            - 'embed': The cell-type embedding array.
        """
        self._init_fileread()
        
        x, y, z = np.unravel_index(idx, shape=self.shape)
        
        embed = self.embeds_vals[x]
        shift = int(self.shifts[y])
        
        chrom = self.coords_chrom[z]
        pos = self.coords_pos[z]
        ref = self.coords_ref[z]
        alt = self.coords_alt[z]

        interval = self._get_window(chrom, pos, shift)

        if self.include_genotypes:
            indiv_id = self.indiv_ids[x]
            _, dna_seq, _, _ = self.get_sample_sequence(interval, indiv_id)
        else:
            dna_seq = self.fasta_extr[interval].upper()

        start = pos - 1
        rel_pos = start - interval.start

        dna_seq_ref, dna_seq_alt = self._prepare_alleles(dna_seq, rel_pos, ref, alt)
        
        ohe_seq_ref = one_hot_encode(dna_seq_ref)
        ohe_seq_alt = one_hot_encode(dna_seq_alt)

        batch = dict(
            ref = ohe_seq_ref.copy(),
            ref_revcomp = np.flip(ohe_seq_ref, [0, 1]).copy(),
            alt = ohe_seq_alt.copy(),
            alt_revcomp = np.flip(ohe_seq_alt, [0, 1]).copy(),
            embed = embed.copy()
        )
        return batch
