import numpy as np

from genome_tools import GenomicInterval, VariantInterval
from vinson.utils.helpers import replace_at
from vinson.utils.data_formatting import VariantVinsonData
from vinson.utils.sequence_utils import one_hot_encode

from .base import BaseSequenceDataset
from .sequence import InferenceSequenceEmbedDataset

from vinson.datasets.utils import PhasedSource, UnphasedSource, TabixConnector
from vinson.utils.sequence_utils import one_hot_encode
from vinson.utils.helpers import detect_genotype_format

import warnings

from vinson.utils.helpers import replace_at
import logging 


# TODO: centralized logging
logger = logging.getLogger(__name__)


# TODO: fix inference similar to sequence.py
class InferenceVariantEmbedDataset(BaseSequenceDataset):
    """
    Prediction-only counterpart of `VariantEmbedDataset`: doesn't parse/require target fields
    (`ref_counts`/`total_counts`/...), and `shift`/`revcomp` are read per-example from `data` (e.g. as
    built by `VariantVinsonData.inference_from_variant`) instead of applied randomly. Requires
    `data.is_inference=True`.

    Doesn't subclass `VariantEmbedDataset` because that class doesn't support inference-mode `data`
    (it asserts `not data.is_inference`).

    Parameters
    ----------
    data : VinsonData
        Must have `is_inference=True`.
    strict_ref_check : bool, default True
        If True, raise if the FASTA reference base at a variant's position doesn't match either the
        variant's `ref` or `alt` allele; if False, log a warning and proceed anyway.
    """

    def __init__(
            self,
            data: VariantVinsonData,
            fasta_file: str,
            genotype_file: str = None,
            seqlen=1344,
            strict_ref_check=True
        ):
        super().__init__(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            seqlen=seqlen
        )
        self.strict_ref_check = strict_ref_check


    def get_source(self):
        connector = self.get_connector()
        if connector is None:
            return UnphasedSource(connector)
        if isinstance(connector, TabixConnector):
            is_phased, _ = detect_genotype_format(self.genotype_file)
        else: # dataframe provided
            is_phased = True

        source_cls = PhasedSource if is_phased else UnphasedSource
        return source_cls(connector)

    def get_sample_sequence( # duplicated from VariantEmbedDataset since this class can't inherit from it
        self,
        interval: GenomicInterval,
        indiv_id: str,
        anchor_variant: VariantInterval = None,
    ):
        """
        Returns:
            tuple: (seq_ref str, seq_alt str)
        """
        seq = self.fasta_extr[interval]
        indiv_id = self._normalize_indiv_id(indiv_id)

        if indiv_id == "":
            return str(seq), str(seq)

        ref_edits, alt_edits = self.source.get_edits(interval, indiv_id, anchor_variant)
        seq_ref = self.source.apply_edits(
            seq,
            interval_start=interval.start,
            edits=ref_edits
        )
        seq_alt = self.source.apply_edits(
            seq,
            interval_start=interval.start,
            edits=alt_edits
        )
        return seq_ref, seq_alt

    def __getitem__(self, i):
        """
        Returns
        -------
        dict
            'ohe_seq_ref', 'ohe_seq_alt' (one-hot ref/alt sequences), 'pos', 'ref', 'alt', 'embed'
            (cell-type embedding), 'chrom', 'shift', 'revcomp'.
        """
        self._init_fileread()
        row = self.data[i]

        chrom = row["chrom"]
        pos = int(row["pos"])  # 1-based
        ref = row["ref"]
        alt = row["alt"]
        sample_id = row["sample_id"]
        shift = int(row.get("shift", 0))
        revcomp = bool(row.get("revcomp", False))

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen // 2)
        if shift != 0:
            interval.shift(shift, inplace=True)

        start = pos - 1
        rel = start - interval.start

        if self.include_genotypes:
            indiv_id = row["indiv_id"]
            anchor_variant = VariantInterval(
                chrom=chrom,
                start=start,
                end=pos,
                ref=ref,
                alt=alt,
            )
            dna_seq_ref, dna_seq_alt = self.get_sample_sequence(
                interval,
                indiv_id,
                anchor_variant
            )
        else:
            base_seq = str(self.fasta_extr[interval])
            dna_seq_ref = replace_at(base_seq, rel, ref)
            dna_seq_alt = replace_at(base_seq, rel, alt)

        base_seq = str(self.fasta_extr[interval]).upper()
       
        fasta_base = base_seq[rel]
        check_var = fasta_base in (ref.upper(), alt.upper())
        if not check_var:
            msg = f"FASTA mismatch at {chrom}:{pos} (rel={rel}): expected {ref}/{alt}, got {fasta_base} in hg38"
            if self.strict_ref_check:
                raise AssertionError(msg)
            else:
                logger.warning(msg)

        dna_seq_ref = replace_at(base_seq, rel, ref)
        dna_seq_alt = replace_at(base_seq, rel, alt)
        if self.strict_ref_check:
            assert check_var, msg
        elif not check_var:
            logger.warning(msg + ". Proceeding anyway (strict_ref_check=False).")

        ohe_ref, _ = self._get_ohe_sequence(dna_seq_ref, revcomp=revcomp)
        ohe_alt, _ = self._get_ohe_sequence(dna_seq_alt, revcomp=revcomp)

        embed = self.get_embedding_vec(sample_id)

        return {
            "ohe_seq_ref": ohe_ref.copy(),
            "ohe_seq_alt": ohe_alt.copy(),
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "embed": embed,
            "chrom": chrom,
            "shift": shift,
            "revcomp": revcomp,
        }


# TODO: remove repetetive methods. Add targets and reuse super().__getitem__()
class VariantEmbedDataset(InferenceVariantEmbedDataset):
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
        data: VariantVinsonData,
        fasta_file: str,
        genotype_file: str = None,
        flip_alleles=False,
        reverse_complement=False,
        random_phase=False,
        jitter=0,
        noise=0,
        seqlen=1344,
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

        self.flip_alleles = flip_alleles

        if self.source is None:
            print('random_phase has no effect when genotype_file is None')
        else:
            self.source.random_phase = random_phase

        assert not data.is_inference, f'Sequency only dataset requires targets to be present in VariantVinsonData data.is_inference={data.is_inference}. Use infererence dataset instead.'

    def get_sample_sequence(
            self,
            interval: GenomicInterval,
            indiv_id: str,
            anchor_variant: VariantInterval=None,
        ):
        """
        Returns:
            tuple: (base_sequence str, variants List[Variant])
        """
        seq = self.fasta_extr[interval]

        indiv_id = self._normalize_indiv_id(indiv_id)

        if indiv_id == "":
            return str(seq), str(seq)
    
        ref_edits, alt_edits = self.source.get_edits(interval, indiv_id, anchor_variant)
        seq_ref = self.source.apply_edits(
            seq,
            interval_start=interval.start,
            edits=ref_edits
        )
        seq_alt = self.source.apply_edits(
            seq,
            interval_start=interval.start,
            edits=alt_edits
        )
        return seq_ref, seq_alt
  
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

        variant = GenomicInterval(chrom, pos - 1, pos)
        interval = variant.resize(self.seqlen)

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        rel_pos = pos - interval.start
        
        # Inject genotypes if genotype files provided
        if self.include_genotypes:
            indiv_id = data_slice['indiv_id']
            anchor_variant = VariantInterval(
                chrom=chrom,
                start=pos-1,
                end=pos,
                ref=ref,
                alt=alt,
            )
        else:
            anchor_variant = None

        dna_seq_ref, dna_seq_alt = self.get_sample_sequence(
            interval, 
            indiv_id,
            anchor_variant
        )

        if len(dna_seq_ref) != self.seqlen or len(dna_seq_alt) != self.seqlen:
            logger.warning(
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
        embed, noise = self.get_embedding_vec(sample_id)

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
