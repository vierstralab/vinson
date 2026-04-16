import anndata as ad

from .sequence import SeqEmbedDataModule

from vinson.datasets.variant import VariantEmbedDataset
from vinson.utils.data_formatting.readers import extract_variant_data_from_anndata
from vinson.utils.data_formatting.adata_utils import get_number_of_train_examples


class SeqEmbedVariantDataModule(SeqEmbedDataModule):
    """
    Variant version of SeqEmbedDataModule.

    Differences:
      - Uses extract_var_data_from_anndata
      - Uses VariantEmbedDataset instead of SequenceEmbedDataset
    """
    def train_dataset(self, epoch=None):
        if epoch is None:
            epoch = self.current_train_epoch

        data = self.train_data[epoch]
        return VariantEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.train_dataset_kwargs,
        )

    def validation_dataset(self):
        data = self.validation_data[self.validation_epoch]  
        return VariantEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.valid_dataset_kwargs,
        )


    @staticmethod
    def get_data(
        full_adata: ad.AnnData,
        suffix: str,
        dhs_split="train",
        sample_split="train",
        pre_jitter=False):
        """Extract variant-level data from AnnData."""
        # TO DO: add in prejitter
        adata = full_adata[
            full_adata.obsm["split_data"] == sample_split,
            full_adata.varm["split_data"] == dhs_split,
        ]
        return extract_variant_data_from_anndata(adata, suffix)
