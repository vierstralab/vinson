import pandas as pd
from typing import List

from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals, filter_df_to_interval
from vinson.utils.data_formatting.readers import parse_genotype_file
from vinson.utils.sequence_utils import get_iupac_char_from_alleles
from vinson.utils.helpers import replace_at


class TabixConnector:
    """Lazy tabix handle. Opened once, reused across intervals."""
    def __init__(self, path):
        self.path = path
        self._extr = None

    def __getitem__(self, interval):
        if self._extr is None:
            self._extr = parse_genotype_file(self.path)
        try:
            return self._extr[interval]
        except ValueError:
            return None


class DataFrameConnector:
    """In-memory genotype df. Same interface as TabixConnector."""
    def __init__(self, df):
        self.df = df

    def __getitem__(self, interval):
        return filter_df_to_interval(self.df, interval=interval)


class _ConnectorSource:
    EXTRA_COLUMNS = ()

    def __init__(self, connector):
        self.connector = connector

    @staticmethod
    def _fix_indiv_id_format(df, indiv_id):
        # defunc method to work with older format files
        if df["indiv_id"].iloc[0].endswith(".bed.gz"):
            indiv_id = f"{indiv_id}.bed.gz"
        return indiv_id

    def _get_indiv_variants_for_interval(self, interval, indiv_id):
        df = self.connector[interval]
        if df is None or df.empty:
            return None

        indiv_id = self._fix_indiv_id_format(df, indiv_id)

        df = df.query(f"indiv_id == '{indiv_id}'")
        if df.empty:
            return []
        return df

    def get_edits(self, interval, indiv_id, anchor=None):
        raise NotImplementedError

    @staticmethod
    def apply_edits(seq, interval_start, edits, strict=True) -> str:
        """Copy seq, apply each VariantInterval edit (writes v.alt at v.start).
        Skips edits outside the window; strict-checks the reference allele."""
        seq = str(seq)
        for v in edits:
            rel = v.start - interval_start
            if rel <= 0 or rel >= len(seq):
                continue
            if strict and seq[rel:rel + len(v.ref)] != v.ref:
                raise AssertionError(f"ref mismatch {v.chrom}:{v.start} expected {v.ref}")
            seq = replace_at(seq, rel, v.alt)
        return seq


class IUPACSource(_ConnectorSource):
    """One sequence with an ambiguity code at every variant site. Phasing-independent."""
    def get_edits(self, interval, indiv_id) -> List[VariantInterval]:
        df = self._get_indiv_variants_for_interval(interval, indiv_id)
        if df is None:
            return []
        iupac = [get_iupac_char_from_alleles(ref, alt)
                 for ref, alt in zip(df["ref"], df["alt"])]
        df['alt'] = iupac
        return df_to_variant_intervals(df)


class _GenotypeSource(_ConnectorSource):
    EXTRA_COLUMNS = ("gt",)

    def _unphased_edits(self, df):
        """ref haplotype: reference, except hom-alt sites (alt on both haplotypes).
        alt haplotype: every alt-bearing variant (het + hom-alt)."""
        if df is None:
            return [], []
        allele_1 = df["gt"].str[0]
        allele_2 = df["gt"].str[2]
        carries_alt = (allele_1 == "1") | (allele_2 == "1")
        is_hom_alt = (allele_1 == "1") & (allele_2 == "1")
        ref_edits = df_to_variant_intervals(df[is_hom_alt], extra_columns=self.EXTRA_COLUMNS)
        alt_edits = df_to_variant_intervals(df[carries_alt], extra_columns=self.EXTRA_COLUMNS)
        return ref_edits, alt_edits


class PhasedSource(_GenotypeSource):
    EXTRA_COLUMNS = ("gt", "phase_set")


    def _get_anchor_variant_with_extras(self, variants_df, interval, reference_variant, indiv_id) -> VariantInterval:
        try:
            #find reference variant in variant
            row: pd.Series = variants_df.set_index(["chrom", "start", "ref", "alt"]).loc[
                (reference_variant.chrom, reference_variant.start, reference_variant.ref, reference_variant.alt)
            ]

        except KeyError:
            #if cannot find variant
            raise ValueError(
                f"Reference variant not found in genotyping file: "
                f"{interval}/{indiv_id}/{reference_variant.start}/{reference_variant.alt}"
            )
        anchor_df = row.to_frame()
        assert len(anchor_df) == 1, f"Genotypes contain not one entry of anchor variant ({len(anchor_df)}): {reference_variant.to_str()}"
        anchor_variant = df_to_variant_intervals(anchor_df, extra_columns=self.EXTRA_COLUMNS)[0]

        return anchor_variant

    def get_edits(self, interval: GenomicInterval, indiv_id, anchor: VariantInterval):
        df = self._variants_in(interval, indiv_id)
        if df is None:
            raise ValueError(f"No variants in the interval {interval.to_ucsc()} for indiv {indiv_id}.")

        anchor_variant = self._get_anchor_variant_with_extras(df, interval, anchor, indiv_id)
        ps = anchor_variant.phase_set
        ps = None if pd.isna(ps) else ps

        if ps is None:   # no phase info -> unphased, no orientation swap
            return self._unphased_edits(df)

        in_ps = df["phase_set"] == ps
        ref_phased, alt_phased = self._phased_edits(df[in_ps])


        # Orient so ref_edits builds seq_ref (anchor reference allele), alt_edits seq_alt.
        if anchor_variant.gt == "1|0":
            ref_phased, alt_phased = alt_phased, ref_phased

        ref_unphased, alt_unphased = self._unphased_edits(df[~in_ps])
        ref_edits = ref_phased + ref_unphased
        alt_edits = alt_phased + alt_unphased
        return ref_edits, alt_edits

    def _phased_edits(self, df):
        """All rows already co-phased with the anchor. Split het by haplotype;
        hom-alt (1|1) goes on both."""
        gt = df["gt"]
        ref_edits = df_to_variant_intervals(df[gt == "1|0"], extra_columns=self.EXTRA_COLUMNS)
        alt_edits = df_to_variant_intervals(df[gt == "0|1"], extra_columns=self.EXTRA_COLUMNS)
        _, hom_alt = self._unphased_edits(df[gt.isin(["1|1"])])
        return ref_edits + hom_alt, alt_edits + hom_alt


class UnphasedSource(_GenotypeSource):
    def get_edits(self, interval: GenomicInterval, indiv_id, anchor=None):
        df = self._variants_in(interval, indiv_id)
        return self._unphased_edits(df)
