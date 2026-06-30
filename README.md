# vinson

vinson is a sequence-to-function modeling framework for chromatin accessibility (DNase I hypersensitivity, DHS) and variant effect prediction.

# Table of contents

- [Installation](#installation)
- [Model architecture](#model-architecture)
- [Config format](#config-format)
- [Loading a trained checkpoint](#loading-a-trained-checkpoint)
- [Training data format](#training-data-format)
- [Training](#training)
- [Prediction](#prediction)
- [Authors](#authors)

# Installation

## Prerequisites
`conda` (or `mamba`), `python >= 3.9`, a CUDA-capable GPU for training/prediction.
<FIXME>

## Install vinson
```bash
git clone https://github.com/vierstralab/vinson
cd vinson
pip install .
```

## Test installation
```bash
python train/dhs/train_dhs.py --help
```

# Model architecture

A DHS model is a convolutional model that consists of  **trunk** and **head**. The trunk accepts a one-hot encoded sequence as input and it's output feeds into a fully connected head that predicts the accessibility. The model either predicts accessibility for many samples at once (`BassetTrunk` or `LegNetTrunk`, see [`vinson/models/sequence/`](src/vinson/models/sequence/)) or predicts a single value depending on the embedding injected into the trunk (`BassetTrunkEmbed` / `LegNetTrunkEmbed`)

The Lightning wrapper (`SequenceOnlyModel` for sequence-only trunks, `SequenceEmbedModel` for embed-conditioned trunks) handles the training/validation loop, loss, and optimizer.

# Config format

Every model is fully described by a single YAML config (see `train/dhs/default_train_dhs.config.yaml` for the schema). During training, each run saves the resolved config, along with a timestamp and the command line used to start the run, to `run_config.yaml`.

Key fields:

- `model_type` — selects a trunk/Lightning-module pair from the registry in [`vinson/from_config.py`](src/vinson/from_config.py):

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

# Loading a trained checkpoint

```python
from vinson.from_config import read_configs, dhs_model_from_config

config = read_configs("<run_dir>/run_config.yaml")
model = dhs_model_from_config(config, checkpoint_path="<run_dir>/checkpoints/last.ckpt").eval()
```

# Training data format

Training datasets are AnnData objects (`.h5ad`) read through [`vinson/utils/data_formatting`](src/vinson/utils/data_formatting). At a minimum, a DHS training example requires `chrom`, `summit`, `class`, `density`, `background`, `read_depth`, and `sample_id`; per-sample cell-type embeddings are looked up by `sample_id` (in `.obsm['motif_embeddings']`). To incorporate individual variants, `indiv_id` should be present in the dataset, along with a tabix file containing the genotype information.

> TODO: document the `VinsonData` container ([`vinson/utils/data_formatting/container.py`](src/vinson/utils/data_formatting/container.py)) and the full AnnData/Zarr schema.

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

- `<anndata_file>` — training AnnData (see [Training data format](#training-data-format))
- `<fasta_file>` — reference genome used to extract and augment sequences on the fly
- `--config` — see [Config format](#config-format) for the contents of the config file
- `--genotype_file` (tabix-indexed, optional) — if provided, each training example's sequence has the individual's genotype (matched by `indiv_id`) injected before one-hot encoding, instead of using the reference allele at every position
- `--outdir` — folder to save the resulting checkpoints and config(s)
- `--run_name` — name for the run; if omitted, a random one is generated
- `--debug` — caps each epoch at 200 batches/device, for a quick smoke test

The script writes to `<outdir>/<run_name>/`: `run_config.yaml`, `checkpoints/*.ckpt`, and CSV training/validation logs.

## SLURM submission

`train/submit_to_sbatch.py` wraps `train_dhs.py` for `sbatch`: it renders `train/dhs/template_submit_dhs.sbatch` with the given arguments and submits it.

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

# Authors

Developed by:

- Sergey Abramov
- Alexandr Boytsov
- Madeline Brannon
- Sergey Bushuev
- Anas Fathul
- Jeff Vierstra

Altius Institute for Biomedical Sciences
