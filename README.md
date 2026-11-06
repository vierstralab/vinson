# vinson

vinson is a package wrapping the core code and processing pipelines for sequence-to-function modeling of chromatin accessibility (DNase I hypersensitivity) and variant effect prediction, along with a cell-type classifier built on the sample embeddings.

It has three main components, each documented in its own README:

- [Accessibility (DHS) sequence-to-function model](docs/accessibility_model.md) — predicts chromatin accessibility from DNA sequence, optionally incorporating the cell-type embedding
- [Variant effect model](docs/variant_model.md) — predicts the allelic effect of a variant on accessibility
- [Cell type classifier](docs/cell_classifier.md) — classifier of a cell-type based on per-sample embeddings


# Table of contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Tutorials](#tutorials)
- [Authors](#authors)

# Repository layout

- [`src/vinson/`](src/vinson/) — the installable package: models, datasets/datamodules, data formatting (`utils/data_formatting/`, see the [`VinsonData`/`VariantVinsonData` containers](docs/accessibility_model.md#trainprediction-data-format)), and model loading from config (`from_config.py`).
- [`train/`](train/) — training entry points and configs for the DHS (`train/dhs/`) and variant (`train/variant/`) models, plus a SLURM submission helper (`submit_to_sbatch.py`).
- [`nextflow/`](nextflow/) — Nextflow pipelines (and their `bin/` scripts) for generating validation data and running prediction at scale.
- [`tutorials/`](tutorials/) — walkthrough notebooks: calling peaks/embeddings, predicting density, interpretation, predicting variants.

# Installation

## Prerequisites
`conda` (or `mamba`), `python >= 3.9`, a CUDA-capable GPU for training/prediction.

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

# Tutorials

Walkthrough notebooks in [`tutorials/`](tutorials/):

1. [Call peaks and generate embeddings](tutorials/1.Call_peaks_and_generate_embeddings.ipynb) — hotspot3 peak calling and building a motif embedding for a new sample
2. [Predict density](tutorials/2.Predict_density.ipynb) — loading a model, predicting accessibility, scoring a BED file with `VinsonData`, plotting predicted tracks
3. [Interpretation](tutorials/3.Interpretation.ipynb) — attribution scores and attribution-weighted motif hits
4. [Predict variant](tutorials/4.Predict_variant.ipynb) — scoring variant effects across cell types

# Authors

Developed by (alphabetically by last name):

- Sergey Abramov
- Alexandr Boytsov
- Madeline Brannon
- Sergey Bushuev
- Anas Fathul
- Dmitry Penzar
- Jeff Vierstra

Altius Institute for Biomedical Sciences
