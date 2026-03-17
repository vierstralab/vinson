import numpy as np

from genome_tools import GenomicInterval, VariantInterval
from genome_tools.data.extractors import TabixExtractor

from vinson.utils.data_formatting import VinsonData
from vinson.utils.sequence_utils import one_hot_encode

from .sequence import BaseSequenceDataset, logger

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
        flip_alleles=True,
        reverse_complement=True,
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
        self.genotype_extr: TabixExtractor = None
        self.fasta_extr: FastaExtractor = None

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

        # chrom, pos, ref, alt, ref_counts, total_counts, bad, lfc, sample_id = (
        #     self.data["chrom"][i],
        #     self.data["pos"][i],
        #     self.data["ref"][i],
        #     self.data["alt"][i],
        #     self.data["ref_counts"][i],
        #     self.data["total_counts"][i],
        #     self.data["BAD"][i],
        #     self.data["logit_es"][i],
        #     self.data["sample_id"][i],
        # )
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

        # FIXME: Decode categorical variables

        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(self.seqlen // 2)

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            interval.shift(shift, inplace=True)

        rel_pos = pos - interval.start
        
        # Inject genotypes if genotype files provided
        #variant interval pos-1 because dataformatting is using end as pos
        if self.include_genotypes:
            indiv_id = data_slice['indiv_id']   
            _, _, dna_seq_ref, dna_seq_alt = self.get_sample_sequence(
                interval, 
                indiv_id,
                reference_variant=VariantInterval(
                    chrom=chrom, start=pos-1, end=pos, ref=ref, alt=alt
                )
            )
        else:
            dna_seq_ref = self.fasta_extr[interval]
            dna_seq_alt = dna_seq_ref[:rel_pos] + alt + dna_seq_ref[rel_pos + 1 :]
        if len(dna_seq_ref) != 1344 or len(dna_seq_alt) != 1344:
            warnings.warn(
            f"[DEBUG WARNING] idx={i}, chrom={chrom}, pos={pos}, "
            f"ref_seq length={len(dna_seq_ref)} alt_len expected seqlen={len(dna_seq_alt)}, relpos {rel_pos}, alt {alt}, ref {ref} indiv {indiv_id}"
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

    def __init__(self, data: VinsonData, **kwargs):
        super().__init__(
            data=data,
            **kwargs
        )
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
        seq = self.fasta_extr[interval]

        # enforce alleles
        center = start - interval.start

        center_base = seq[center].upper()
        # optional sanity check
        assert center_base in (ref, alt), f"Alleles mismatch at {chrom}:{start} for sample {sample_id}: expected {ref}/{alt}, got {seq[center]}"

        seq_ref = replace_at(seq, center, ref)
        seq_alt = replace_at(seq, center, alt)

        ohe_ref = one_hot_encode(seq_ref)
        ohe_alt = one_hot_encode(seq_alt)

        embed = self.get_embedding_vec(sample_id)

        return {
            "ohe_seq_ref": ohe_ref,
            "ohe_seq_alt": ohe_alt,
            "center_seq": center_base,
            "ref": ref,
            "alt": alt,
            "embed": embed,
            "chrom": chrom,
            "start": start,
        }