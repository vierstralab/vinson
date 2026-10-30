import numpy as np

from genome_tools import GenomicInterval, VariantInterval
from vinson.utils.data_formatting import VariantVinsonData

from .base import BaseSequenceDataset

from vinson.datasets.utils import PhasedSource, UnphasedSource, TabixConnector
from vinson.utils.helpers import apply_edits, detect_genotype_format
import logging


logger = logging.getLogger(__name__)


class InferenceVariantEmbedDataset(BaseSequenceDataset):
    """
    Prediction-only counterpart of `VariantEmbedDataset`: doesn't parse/require target fields
    (`ref_counts`/`total_counts`/...), and `shift`/`revcomp` are read per-example from `data` (e.g. asso whi
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
            strict_ref_check=True,
            **kwargs
        ):
        super().__init__(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            seqlen=seqlen,
            **kwargs
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

    def _parse_position(self, data_dict: dict):
        chrom = data_dict['chrom']
        pos = int(data_dict['pos'])  # 1-based
        ref = data_dict['ref']
        alt = data_dict['alt']

        shift = data_dict.get('shift')
        shift = int(shift) if shift is not None else None

        anchor_variant = VariantInterval(chrom, pos - 1, pos, ref, alt)
        interval, shift = self._get_window(chrom, pos - 1, shift=shift)
        return interval, anchor_variant, shift

    def _parse_sequence(self, data_dict: dict, interval: GenomicInterval, anchor_variant: VariantInterval):
        if self.include_genotypes:
            indiv_id = data_dict['indiv_id']
        else:
            indiv_id = ""
        self._init_fileread()

        dna_seq_ref, dna_seq_alt = self.get_sample_sequence(interval, indiv_id, anchor_variant)

        if 'revcomp' in data_dict:
            revcomp = data_dict['revcomp']
        else:
            revcomp = None

        # Resolve the revcomp decision once (random if not given) and apply it to both alleles.
        ohe_seq_ref, revcomp = self._get_ohe_sequence(dna_seq_ref, revcomp=revcomp)
        ohe_seq_alt, _ = self._get_ohe_sequence(dna_seq_alt, revcomp=revcomp)

        return ohe_seq_ref.copy(), ohe_seq_alt.copy(), revcomp

    def get_sample_sequence(
            self,
            interval: GenomicInterval,
            indiv_id: str,
            anchor_variant: VariantInterval,
        ):
        """
        Returns:
            tuple: (seq_ref str, seq_alt str)
        """
        seq = self.fasta_extr[interval]

        indiv_id = self._normalize_indiv_id(indiv_id)

        if indiv_id == "":
            ref_edits = [
                VariantInterval(
                    anchor_variant.chrom,
                    anchor_variant.start,
                    anchor_variant.end,
                    anchor_variant.ref,
                    anchor_variant.ref,
                )
            ]
            alt_edits = [anchor_variant]
        else:
            ref_edits, alt_edits = self.source.get_edits(interval, indiv_id, anchor_variant)

        seq_ref = apply_edits(seq, interval=interval, edits=ref_edits, strict=self.strict_ref_check)
        seq_alt = apply_edits(seq, interval=interval, edits=alt_edits, strict=self.strict_ref_check)
        return seq_ref, seq_alt

    def __getitem__(self, i):
        """
        Returns
        -------
        dict
            'ohe_seq_ref', 'ohe_seq_alt' (one-hot ref/alt sequences), 'pos', 'ref', 'alt', 'embed'
            (cell-type embedding), 'chrom', 'shift', 'revcomp', 'sample_id'.
        """
        data_slice = self._slice_data(i)

        interval, anchor_variant, shift = self._parse_position(data_slice)
        ohe_seq_ref, ohe_seq_alt, revcomp = self._parse_sequence(data_slice, interval, anchor_variant)

        sample_id = data_slice['sample_id']
        embed, _ = self.get_embedding_vec(sample_id)

        return {
            "ohe_seq_ref": ohe_seq_ref,
            "ohe_seq_alt": ohe_seq_alt,
            "embed": embed.copy(),
            "pos": anchor_variant.pos,
            "ref": anchor_variant.ref,
            "alt": anchor_variant.alt,
            "chrom": anchor_variant.chrom,
            "shift": shift,
            "revcomp": revcomp,
            "sample_id": sample_id,
        }


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
    strict_ref_check : bool, default True
        If True, raise if the FASTA reference base at a variant's position doesn't match either the
        variant's `ref` or `alt` allele; if False, log a warning and proceed anyway.

    Notes
    -----
    The terminology "ref" vs. "alt" is a bit of a misnomer, as it is really
    haplotype 1 vs. haplotype 2. We call it "ref" vs. "alt" because the
    variant effect is always measured against the reference genome allele.
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
        strict_ref_check=True,
    ):
        super().__init__(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            reverse_complement=reverse_complement,
            jitter=jitter,
            noise=noise,
            seqlen=seqlen,
            strict_ref_check=strict_ref_check,
        )

        self.flip_alleles = flip_alleles

        if not self.include_genotypes:
            logger.warning('random_phase has no effect when genotype_file is None')
        else:
            self.source.random_phase = random_phase

        assert not data.is_inference, f'Sequence dataset requires targets to be present in VariantVinsonData data.is_inference={data.is_inference}. Use infererence dataset instead.'

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
                - 'ohe_seq_ref': np.ndarray, one-hot encoded reference allele sequence
                - 'ohe_seq_alt': np.ndarray, one-hot encoded alternate allele sequence
                - 'embed': np.ndarray, cell-type/state embedding
                - 'ref_counts': float, reference allele counts
                - 'total_counts': float, total counts (reference + alternate)
                - 'bad_score': float, BAD score for the variant
                - 'lfc': float, log fold change (in natural log units)
                - 'sample_id': str, sample identifier
                - 'weight': float, sample weight (default 1.0)
        """
        result = super().__getitem__(i)

        # targets
        data_slice = self._slice_data(i)
        ref_counts = data_slice['ref_counts']
        total_counts = data_slice['total_counts']
        bad = data_slice['BAD']
        lfc = data_slice['logit_es']

        # Flip reference and alternative alleles in input
        # for additional regularization
        if self.flip_alleles and np.random.choice(2) == 1:
            result['ohe_seq_ref'], result['ohe_seq_alt'] = result['ohe_seq_alt'], result['ohe_seq_ref']
            ref_counts = total_counts - ref_counts
            lfc = -1 * lfc

        result.update(
            {
                "ref_counts": ref_counts,
                "total_counts": total_counts,
                "bad_score": bad,
                "lfc": lfc, # CHANGE LFC to be log2 IN the model.py
                "weight": 1.0,
            }
        )
        return result


class VariantEmbedDensityDataset(VariantEmbedDataset):
    """
    `VariantEmbedDataset` with DNase density targets, for training `VariantEmbedDensityModel`.

    Returns everything `VariantEmbedDataset` does (ref/alt sequences, embedding, allelic
    imbalance targets) plus the DNase density at the center of the window, with the
    background and read depth needed to convert density to counts in the model's loss.

    Expects pre-jittered data: density is measured at the center of the stored window, so
    shifting the window here would misalign the sequence and its target. `jitter` must be 0.

    Parameters
    ----------
    *args, **kwargs
        Passed to `VariantEmbedDataset` (`data`, `fasta_file`, `genotype_file`,
        `flip_alleles`, `reverse_complement`, `random_phase`, `noise`, `seqlen`,
        `strict_ref_check`). `data` must include `density`, `background` and `read_depth`.
    clip_density : float, default 20
        Maximum value to clip density.
    min_bg : float, default 0.1
        Minimum value to clip background.

    Raises
    ------
    ValueError
        If `jitter` is nonzero.

    Notes
    -----
    Density is not swapped when `flip_alleles` swaps ref and alt: the model predicts density
    as the mean of the ref and alt predictions, so the target is the same either way.

    Returns (from `__getitem__`)
    ----------------------------
    dict
        Keys from `VariantEmbedDataset.__getitem__`, plus:
            - 'density': float, clipped DNase density at the window center
            - 'bg': float, clipped background
            - 'read_depth': float, sample read depth
    """
    def __init__(self, *args, clip_density=20, min_bg=0.1, **kwargs):
        if kwargs.get("jitter", 0):
            raise ValueError("Data is pre-jittered; set jitter=0")
        super().__init__(*args, **kwargs)
        self.clip_density = clip_density
        self.min_bg = min_bg

    def __getitem__(self, i):
        result = super().__getitem__(i)
        d = self._slice_data(i)
        result.update({
            "density": np.float32(np.clip(d["density"], None, self.clip_density)),
            "bg": np.float32(np.clip(d["background"], self.min_bg, None)),
            "read_depth": np.float32(d["read_depth"]),
        })
        return result

