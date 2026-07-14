# Accessibility (DHS) sequence-to-function model

Part of [vinson](../README.md). Predicts chromatin accessibility directly from DNA sequence.

# Table of contents

- [Model architecture](#model-architecture)
- [Loading a trained checkpoint](#loading-a-trained-checkpoint)
- [Config format](#config-format)
- [Train/prediction data format](#trainprediction-data-format)
- [Training](#training)
- [Prediction](#prediction)

# Model architecture

A DHS model is a convolutional model that consists of  **trunk** and **head**. The trunk accepts a one-hot encoded sequence as input and it's output feeds into a fully connected head that predicts the accessibility. The model either predicts accessibility for many samples at once (`BassetTrunk` or `LegNetTrunk`, see [`vinson/models/sequence/`](../src/vinson/models/sequence/)) or predicts a single value depending on the embedding injected into the trunk (`BassetTrunkEmbed` / `LegNetTrunkEmbed`)

The Lightning wrapper (`SequenceOnlyModel` for sequence-only trunks, `SequenceEmbedModel` for embed-conditioned trunks) handles the training/validation loop, loss, and optimizer.

# Loading a trained checkpoint

```python
from vinson.from_config import read_configs, dhs_model_from_config

config = read_configs("<run_dir>/run_config.yaml")
model = dhs_model_from_config(config, checkpoint_path="<run_dir>/checkpoints/last.ckpt").eval()
```

# Config format

Every model is fully described by a single YAML config (see `train/dhs/default_train_dhs.config.yaml` for the schema). During training, each run saves the resolved config, along with a timestamp and the command line used to start the run, to `run_config.yaml`.

Key fields:

- `model_type` — selects a trunk/Lightning-module pair from the registry in [`vinson/from_config.py`](../src/vinson/from_config.py):

| `model_type` | Trunk | Lightning module |
| --- | --- | --- |
| `basset` | `BassetTrunk` — sequence only | `SequenceOnlyModel` |
| `legnet` | `LegNetTrunk` — sequence only | `SequenceOnlyModel` |
| `basset_embed` | `BassetTrunkEmbed` — sequence + cell-type embedding | `SequenceEmbedModel` |
| `legnet_embed` | `LegNetTrunkEmbed` — sequence + cell-type embedding | `SequenceEmbedModel` |

- `model_arch` — `trunk` and `head` configuration. For `*_embed` model types it also requires a `cell_embed` configuration (kwargs for the fully connected block that injects the cell-type embedding into the model).
- `hparams` — `batch_size`, `optimizer_kwargs.lr`, `lr_scheduler`, …
- `train_augmentation_kwargs` / `validation_augmentation_kwargs` — data augmentation applied per example:
  - `reverse_complement` (bool) — randomly reverse-complement the input sequence
  - `jitter` (int) — randomly shift the genomic window by up to `jitter` bp
  - `noise` (float) — std. dev. of Gaussian noise added to the cell-type embedding
- `data_params` — applied to both train/validation:
  - `clip_density` — clips the target accessibility density to this max value
  - `min_bg` — floors the background signal at this value (avoids NaNs in log)
  - `negatives_weight` — loss weight multiplier applied to negative (non-accessible) examples
- `logging_params` — `val_check_interval`, `logger_type`

# Train/prediction data format

`SequenceDataset` and `SequenceEmbedDataset` ([`vinson/datasets/sequence.py`](../src/vinson/datasets/sequence.py)) wrap the data used to train/predict with the model, extracting one-hot encoded sequence windows around each DHS on the fly from a `fasta_file`.

```python
SequenceEmbedDataset(data=data, fasta_file="<fasta_file>") # optionally can provide genotype_file
```

`data` is a `VinsonData` container ([`vinson/utils/data_formatting/container.py`](../src/vinson/utils/data_formatting/container.py)) — a lightweight wrapper around parallel arrays, one entry per training/prediction example (`chrom`, `summit`, `class`, `density`, `background`, `read_depth`, `sample_id`, optionally `indiv_id`), plus an optional per-sample `embeddings_df` (cell-type embeddings, indexed by `sample_id`). Categorical fields (e.g. `chrom`, `sample_id`) are integer-encoded on construction; the original values are recoverable via `.decode(key)` or `.to_df()`. `VinsonData` is normally built via one of the `extract_data_from_*` helpers below, rather than constructed directly.

## From train Anndata
Training datasets are a single AnnData object (`.h5ad`) per training run, read through [`vinson/utils/data_formatting`](../src/vinson/utils/data_formatting), with the following schema:

- `var`: `#chr`, `start`, `dhs_summit` — DHS coordinates.
- `uns.epoch_names` — list of epoch suffixes (e.g. `epoch_1`, `epoch_2`, …) used to look up that epoch's layers, `{layer}.{epoch_name}`.
- `uns.n_training_examples` — total number of training examples across all epochs, cached so it doesn't need to be recomputed by summing nonzero entries over every epoch layer (e.g. to compute total steps for a OneCycle LR schedule).
- `varm.dhs_weight` — per-DHS loss weight, multiplied by the `negatives_weight` multiplier (see [Config format](#config-format)) to give each example's final training loss weight; can be used to give higher weight to cell-selective DHSs.
- `obsm.split_data` / `varm.split_data` — `train`/`val` labels along the sample (`obsm`) and DHS (`varm`) axes, e.g. holding out all DHSs on chr9 and chr22 for validation.
- `obsm.motif_embeddings` — DataFrame indexed by `sample_id`, one column per motif/TF, holding the normalized per-sample motif-proportion embedding.
- `layers`: `binary`, plus `class.<epoch_name>`, `density.<epoch_name>`, `mean_bg_agg_cutcounts.<epoch_name>` for each name in `uns.epoch_names` (e.g. `class.epoch_1`, `class.epoch_2`, …); if that epoch was extracted with `pre_jitter=True`, `offsets.<epoch_name>` is present as well.
  - `binary` — whether the individual sample's peak call overlaps the DHS
  - `density` — cuts density in the 151 bp window around the summit, normalized to library size (`not_normalized_density * 1e6 / total_nuclear_reads`)
  - `mean_bg_agg_cutcounts` — the background cuts density, not normalized to library size

The per-epoch layers are stored as sparse (CSR) matrices containing only the examples selected for that epoch.

To incorporate individual variants, `obsm.indiv_id` — a `pd.DataFrame` with an `indiv_id` column — should be present in the dataset, along with a tabix file containing the genotype information.

`extract_data_from_train_anndata(train_adata, suffix, pre_jitter=False)` ([`vinson/utils/data_formatting/readers.py`](../src/vinson/utils/data_formatting/readers.py)) builds a `VinsonData` from a training AnnData for a chosen epoch. `suffix` selects which epoch's layers to read (layers are named `{layer}.{suffix}`, e.g. `class.epoch_1`). With `pre_jitter=False`, each example's `summit` is left at the DHS's stored `dhs_summit` and jitter is instead applied on the fly during training (see `train_augmentation_kwargs.jitter` in [Config format](#config-format)). With `pre_jitter=True`, the epoch's density was already computed at a jittered offset from the summit, so the per-example offsets stored in the `offsets.<suffix>` layer are added to `summit`, yielding the actual genomic position at which density was extracted.

## From h5 file
`extract_data_from_h5(h5_file, ref_adata, is_variant=False)` reads a previously-written `.h5` file (written by `VinsonData.write_h5`) and pairs it with cell-type embeddings looked up from `ref_adata.obsm['motif_embeddings']` — `ref_adata` is an `ad.AnnData` object, either a backed reference AnnData or a training AnnData, as long as it has `.obsm['motif_embeddings']`.

The `.h5` file has one top-level dataset per `VinsonData` field it was written from — the same fields as the training AnnData schema above: `chrom`, `summit`, `class`, `density`, `background`, `read_depth`, `sample_id`, `dhs_id`, and optionally `indiv_id` / `dhs_weight`.

```python
from vinson.utils.data_formatting import extract_data_from_h5

data = extract_data_from_h5("<h5_file>", ref_adata="<adata_with_sample_embeds>", is_variant=False)
```

## From anndata file
`extract_data_from_backed_anndata(backed_anndata, dhs_ids=None, sample_ids=None, use_sample_peaks=False, extra_layers=())` builds a `VinsonData` directly from a (Zarr-)backed reference AnnData — used for prediction/validation rather than training — by broadcasting every requested `sample_id` x `dhs_id` pair into one row per example. `use_sample_peaks=True` filters to DHSs called as peaks (`class == 1`) in each sample. `extract_data_from_backed_anndata_wide` does the same but skips embeddings and keeps one row per DHS with samples as columns, for a wide per-DHS prediction layout (e.g. the cell-selectivity prediction described under [Prediction](#prediction)) instead of the flattened per-example format.

## Useful post-processing functions

[`vinson/postprocessing/utils.py`](../src/vinson/postprocessing/utils.py):
- `annotate_eval_dataset_with_layers(eval_dataset, annotate_counts=False, annotate_log_density=False, **kwargs)` — annotates a predictions dataframe (`VinsonData.to_df()` with added prediction column `pred_corrected_density`) with background-corrected density/count columns (and optionally their log-transforms) for comparing predictions against observed data (including or excluding the background).

[`vinson/postprocessing/interpretation.py`](../src/vinson/postprocessing/interpretation.py) — DeepLIFT/SHAP-based sequence attribution (`deep_lift_shap`, built on `tangermeme`), for identifying which bases/motifs drive a model's prediction for a given input sequence.

# Training

## Python API

The model is a Lightning `LightningModule`, so it trains with a standard Lightning `Trainer`:

```python
import lightning as L
from vinson.from_config import read_configs, dhs_model_from_config, datamodule_from_config

config = read_configs("<model_config>")
model = dhs_model_from_config(config)

datamodule = datamodule_from_config(
    config,
    anndata_file="<train_anndata_file>",
    fasta_file="<fasta_file>",
    genotype_file=None,
)

trainer = L.Trainer(max_epochs=20, accelerator="gpu", devices=1)
trainer.fit(model, datamodule=datamodule)
```

Alternatively, build the dataset directly from a `VinsonData` object — e.g. loaded from an `.h5` file — and pass plain dataloaders to `trainer.fit` instead of a datamodule:

```python
from torch.utils.data import DataLoader
from vinson.utils.data_formatting import extract_data_from_h5
from vinson.datasets.sequence import SequenceEmbedDataset

data = extract_data_from_h5("<h5_file>", ref_adata="<adata_with_sample_embeds>", is_variant=False)
dataset = SequenceEmbedDataset(data=data, fasta_file="<fasta_file>")
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

trainer.fit(model, train_dataloaders=train_loader)
```

## Training script

`train/dhs/train_dhs.py` wraps the above into a CLI.

```bash
python train/dhs/train_dhs.py <anndata_file> <fasta_file> \
    --config train/dhs/default_train_dhs.config.yaml \
    --genotype_file <genotype.bed.gz> \
    --outdir <outdir> \
    --devices 4
```

- `<anndata_file>` — training AnnData (see [Train/prediction data format](#trainprediction-data-format))
- `<fasta_file>` — reference genome used to extract and augment sequences on the fly
- `--config` — see [Config format](#config-format) for the contents of the config file
- `--genotype_file` (tabix-indexed, optional) — if provided, each training example's sequence has the individual's genotype (matched by `indiv_id`) injected before one-hot encoding, instead of using the reference allele at every position
- `--outdir` — folder to save the resulting checkpoints and config(s)
- `--run_name` — name for the run; if omitted, a random one is generated
- `--debug` — caps each epoch at 200 batches/device, for a quick smoke test

The script writes to `<outdir>/<run_name>/`: `run_config.yaml`, `checkpoints/*.ckpt`, and CSV training/validation logs.

## SLURM submission

`train/submit_to_sbatch.py` wraps `train_dhs.py` for SLURM-enabled clusters. It renders `train/dhs/template_submit_dhs.sbatch` with the given arguments and submits it.

```bash
python train/submit_to_sbatch.py --model_type dhs <anndata_file> <fasta_file> <outdir> \
    --genotype_file <genotype.bed.gz> \
    --config train/dhs/default_train_dhs.config.yaml \
    --gpus_per_node 4
```

Additional arguments controlling the job: `--run_name`, `--gpus_per_node`, `--cpus_per_gpu`, `--mem`, `--nodelist`, `--preset` (cluster-specific node presets), `--checkpoint`, `--epochs`, `--debug`.

# Prediction

Prediction is run through Nextflow, driven by a tab-separated samplesheet (see `nextflow/test_meta/*.tsv` for examples) with one row per sample/DHS to predict:

| column | description |
| --- | --- |
| `prefix` | output file prefix |
| `sample_id` / `dhs_id` | sample(s) or DHS(s) to generate validation data for |
| `zarr_anndata` | reference AnnData (Zarr) with sequence regions |
| `fasta_file` | reference FASTA |
| `checkpoint` | trained model checkpoint (`.ckpt`) |
| `model_config` | training `run_config.yaml` for that checkpoint |
| `genotype_file` | (optional) tabix-indexed genotype file, for sample-specific alleles |

```bash
nextflow run nextflow/generate_data.nf \
    -profile Altius \
    --validation_samples_file nextflow/test_meta/test_per_sample_meta.tsv \
    --outdir <outdir>
```

This generates per-sample/per-DHS validation datasets, runs the model, and annotates results with the input metadata. To predict and plot cell-selectivity for a set of DHSs instead of a sample list, use the `dhsValidation` entrypoint:
```bash
nextflow run nextflow/generate_data.nf -entry dhsValidation \
    -profile Altius \
    --validation_dhs_file nextflow/test_meta/top_cell_selective_dhs_meta.tsv \
    --outdir <outdir>
```

Writes to `<outdir>/predictions/<prefix>/`: `<prefix>.npy` (raw predictions) and, for cell-selective runs, `<prefix>*.pdf` plots.
