import anndata as ad
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals
from genome_tools.data.extractors import FastaExtractor, TabixExtractor

def get_data(self, suffix, dhs_split="train", sample_split="train"):
        """Extract variant-level data from AnnData."""
        adata = self.adata

        exclude_ids = []
        if exclude_ids and "indiv_id" in adata.obsm:
            indiv_ids = adata.obsm["indiv_id"]  # assumes (n_obs, 1)
            keep_mask = ~np.isin(indiv_ids, exclude_ids)
            adata = adata[keep_mask].copy()  # only keep allowed samples

            
        return extract_variant_data_from_anndata(adata, suffix)

def extract_variant_data_from_anndata(train_adata: ad.AnnData, suffix: str):
    """
    Convert AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    
    layers = {"ref_counts": None, "total_counts": None, "BAD": None, "logit_es": None, 'FDR_sample':None}
    row_idx, col_idx = update_layers_dict_var(layers, train_adata, suffix)

    data = {
        'chrom': train_adata.var['#chr'].values[col_idx],
        'pos': train_adata.var['start'].values[col_idx],
        'ref': train_adata.var['ref'].values[col_idx],
        'alt': train_adata.var['alt'].values[col_idx],
        'sample_id': train_adata.obs_names[row_idx],
        'ref_counts': layers['ref_counts'].data,
        'total_counts': layers['total_counts'].data,
        'BAD': layers['BAD'].data,
        'logit_es': layers['logit_es'].data,
        'FDR_sample': layers['FDR_sample'].data,
    }

    if 'indiv_id' in train_adata.obsm:
        data['indiv_id'] = get_indiv_id_info(train_adata, row_idx)

    return data

def sanitize_data(data: dict, is_variant=False) -> dict:
    """Ensure that all data arrays are contiguous and correct dtype."""
    
    if is_variant:
        data_keys = {
            "chrom": np.str_,
            "pos": np.int32,
            "ref": np.str_,
            "alt": np.str_,
            "ref_counts": np.float32,
            "total_counts": np.float32,
            "BAD": np.float32,
            "sample_id": np.str_,
            "logit_es": np.float32,
            "FDR_sample": np.float32,
        }
    else:
        data_keys = {
            "read_depth": np.float32,
            "sample_id": np.str_,
            "dhs_id": np.str_,
            "chrom": np.str_,
            "summit": np.int32,
            "background": np.float32,
            "class": np.int8,
            "density": np.float32,
        }

    optional_keys = {
        "indiv_id": np.str_,
        "dhs_weight": np.float32,
    }

    keys = {
        **data_keys,
        **{k: v for k, v in optional_keys.items() if k in data},
    }

    for key, dtype in keys.items():
        # Convert ANY incoming structure (Index, Series, list) into numpy array
        arr = np.asarray(data[key], dtype=object)

        # if dtype == np.str_:
        #     mask = pd.isna(arr) | np.isin(arr, ['None', 'nan'])
        #     arr[mask] = ''
        if dtype is str:
            # Force Python strings, preserve exact chrom names
            arr = np.array(
                ["" if pd.isna(x) or x in ("None", "nan") else str(x) for x in arr],
                dtype=object,
            )
        else:
            arr = arr.astype(dtype, copy=False)

        data[key] = np.ascontiguousarray(arr.astype(dtype))

    if 'background' in data:
        data['background'] = np.nan_to_num(data['background'])

    return data

def get_indiv_id_info(train_adata: ad.AnnData, row_idx: np.ndarray):
    #return train_adata.obsm['indiv_id'].values[row_idx]
    return train_adata.obsm['indiv_id'][row_idx]


def update_layers_dict_var(layers: dict, train_adata: ad.AnnData, suffix: str):
    assert len(layers) > 0, "Must provide at least one layer to extract"
    for layer_name in layers:
        epoch_layer_name = f"{layer_name}.{suffix}"
        layers[layer_name] = train_adata.layers[epoch_layer_name].tocoo()
    
    class_coo = layers["logit_es"]
    row_idx, col_idx = class_coo.row, class_coo.col
    return row_idx, col_idx


###############################################
# LOAD DATA
###############################################

variant_file = "/net/seq/data2/projects/mbrannon/vinson/variant_train_adata_epochs_balanced_noambig.h5ad"
adata = ad.read_h5ad(variant_file)
d = extract_variant_data_from_anndata(adata, 'epoch_1')
data = sanitize_data(d, is_variant=True)
genotype_file = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/phasing/output/all_phased.bed.gz"
fasta_file = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"

fasta_extr = FastaExtractor(fasta_file)

genotype_extr = TabixExtractor(
    genotype_file,
    skiprows=1,
    columns=["chrom", "start", "end", "ref", "alt", "indiv_id", "gt", "phase_block"],
    na_values={"phase_block": "."},
)

###############################################
# PREPARE OUTPUT
###############################################

# ambiguous_records = []   # collect (chrom, pos, indiv_id, reason)
# n_variants = adata.n_vars
###############################################
# MULTI-EPOCH AMBIGUUOUS DETECTION (FIXED)
###############################################

epochs = ["epoch_1", "epoch_2", "epoch_3"]

all_ambiguous_records = []
all_ambig_var_idx = set()   # <-- GLOBAL VAR INDICES TO REMOVE

skip_ids = ()

for epoch in epochs:
    print(f"\n==============================")
    print(f"Processing {epoch}")
    print(f"==============================")

    # extract per-epoch data
    d = extract_variant_data_from_anndata(adata, epoch)
    data = sanitize_data(d, is_variant=True)

    # get the col_idx mapping for this epoch
    layers = {"logit_es": None}
    _, col_idx = update_layers_dict_var(layers, adata, epoch)

    ambiguous_records = []

    n_entries = len(data["chrom"])
    print(f"Nonzero variant entries this epoch: {n_entries}")

    ###############################################
    # LOOP THROUGH *NONZERO VARIANT ENTRIES*
    ###############################################

    for i in tqdm(range(n_entries), desc=f"Checking variants ({epoch})"):

        chrom, pos, ref, alt = (
            data["chrom"][i],
            data["pos"][i],
            data["ref"][i],
            data["alt"][i],
        )
        indiv_id = data["indiv_id"][i]

        # GLOBAL VAR INDEX IN AnnData
        var_idx = col_idx[i]

        # Skip excluded
        if indiv_id in skip_ids:
            continue

        # Missing indiv_id
        if pd.isna(indiv_id) or indiv_id in ("None", ""):
            ambiguous_records.append((chrom, pos, indiv_id, epoch, "missing_indiv_id"))
            all_ambig_var_idx.add(var_idx)
            continue

        # Interval for lookup
        variant = GenomicInterval(chrom, pos, pos)
        interval = variant.widen(1344 // 2)

        # Fetch genotype records overlapping interval
        variants = genotype_extr[interval]

        # Normalize indiv_id value
        if variants["indiv_id"].str.endswith(".bed.gz").any():
            indiv_key = f"{indiv_id}.bed.gz"
        else:
            indiv_key = indiv_id

        variants = variants[variants["indiv_id"] == indiv_key]

        # Handle phase column
        if "phase_set" not in variants.columns:
            if "phase_block" in variants.columns:
                variants = variants.rename(columns={"phase_block": "phase_set"})
            else:
                variants["phase_set"] = None

        # ---- Step 1: Must find the exact reference variant ----
        try:
            _ = (
                variants
                .set_index(["chrom", "start", "ref", "alt"])
                .loc[(chrom, pos, ref, alt)]
            )
        except KeyError:
            ambiguous_records.append((chrom, pos, indiv_id, epoch, "missing_in_tabix"))
            all_ambig_var_idx.add(var_idx)
            continue

        # ---- Step 2: Convert df to VariantIntervals ----
        variants_vi = df_to_variant_intervals(variants, extra_columns=("gt", "phase_set"))

        # ---- Step 3: Group by position to detect ambiguity ----
        variants_by_pos = {}
        for v in variants_vi:
            variants_by_pos.setdefault(v.start, []).append(v)

        for vpos, vars_here in variants_by_pos.items():
            if len(vars_here) > 1:
                ambiguous_records.append((chrom, vpos, indiv_id, epoch, "multiple_variants"))
                all_ambig_var_idx.add(var_idx)
                break

    ###############################################
    # SUMMARY FOR THIS EPOCH
    ###############################################

    ambig_df_epoch = pd.DataFrame(
        ambiguous_records,
        columns=["chrom", "pos", "indiv_id", "epoch", "reason"]
    )

    print(f"\nEpoch {epoch}:")
    print(f"  Ambiguous entries: {len(ambig_df_epoch)}")
    print(f"  Unique ambiguous sites: {ambig_df_epoch[['chrom','pos']].drop_duplicates().shape[0]}")
    print(f"  Unique AnnData vars flagged: {len(set(col_idx[i] for i in range(len(col_idx)) if col_idx[i] in all_ambig_var_idx))}")

    all_ambiguous_records.extend(ambiguous_records)


###############################################
# BUILD GLOBAL FILTER MASK (ANY EPOCH AMBIGUOUS)
###############################################

print("\n==============================")
print("Building global filter mask")
print("==============================")

ambig_df = pd.DataFrame(
    all_ambiguous_records,
    columns=["chrom", "pos", "indiv_id", "epoch", "reason"]
)

ambig_df.to_csv(
    "/net/seq/data2/projects/mbrannon/ambiguous_variants_all_epochs.tsv",
    sep="\t",
    index=False,
)

print(f"\nSaved ambiguous list: ambiguous_variants_all_epochs.tsv")
print(f"Total ambiguous entries (all epochs): {len(ambig_df)}")
print(f"Total unique ambiguous AnnData vars: {len(all_ambig_var_idx)}")

# Build mask over *all* vars
mask = np.ones(adata.n_vars, dtype=bool)
mask[list(all_ambig_var_idx)] = False

print(f"\nOriginal variants: {adata.n_vars}")
print(f"Filtered variants: {mask.sum()}")
print(f"Removed variants: {adata.n_vars - mask.sum()}")

###############################################
# SAVE FILTERED ANNDATA
###############################################

filtered_adata = adata[:, mask].copy()
filtered_adata.write_h5ad(
    "/net/seq/data2/projects/mbrannon/adata.no_ambiguous_all_epochs_wfdr.h5ad"
)

print("\nFiltered file written to:")
print("  /net/seq/data2/projects/mbrannon/adata.no_ambiguous_all_epochs_wfdr.h5ad")
print("Done.")

