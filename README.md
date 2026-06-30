# vinson

vinson is a sequence-to-function modeling framework for chromatin accessibility (DNase I hypersensitivity, DHS) and variant effect prediction.

It has three main components, each documented in its own README:

- [Cell type classifier](docs/cell_classifier.md) — classifier of a cell-type based on per-sample embeddings
- [Accessibility (DHS) sequence-to-function model](docs/accessibility_model.md) — predicts chromatin accessibility from DNA sequence, optionally incorporating the cell-type embedding
- [Variant effect model](docs/variant_model.md) — predicts the allelic effect of a variant on accessibility


# Table of contents

- [Installation](#installation)
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

# Authors

Developed by (alphabetically by last name):

- Sergey Abramov
- Alexandr Boytsov
- Madeline Brannon
- Sergey Bushuev
- Anas Fathul
- Jeff Vierstra

Altius Institute for Biomedical Sciences
