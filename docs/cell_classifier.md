# Cell type classifier

Part of [vinson](../README.md). Classifies cell type (and other sample metadata, e.g. pathological state) from a per-sample motif-proportion embedding, rather than from DNA sequence - a simple MLP over the embedding.

# Table of contents

- [Model architecture](#model-architecture)
- [Loading a trained checkpoint](#loading-a-trained-checkpoint)
- [Config format](#config-format)
- [Training data format](#training-data-format)
- [Training](#training)

# Model architecture

`CellClassifierModel` ([`vinson/models/cell_classifier.py`](../src/vinson/models/cell_classifier.py)) is a multi-head classifier that consists of a shared MLP trunk (`MLPBlock`, see [`vinson/models/shared.py`](../src/vinson/models/shared.py)) and one linear head per task that maps that representation to per-class logits. This architecture supports training on multiple label columns at once (e.g. `cell_type` and `pathological_state`) with a single shared trunk. Each head is trained with cross-entropy loss; the total loss is the sum across heads.

The Lightning wrapper handles the training/validation loop and optimizer, consistent with the other models in vinson.

# Loading a trained checkpoint

```python
from vinson.from_config import read_configs, classifier_model_from_config

config = read_configs("<run_dir>/run_config.yaml")
model = classifier_model_from_config(config, checkpoint_path="<run_dir>/checkpoints/last.ckpt").eval()
```

# Config format

- `model_type` — must be `classifier` to select `CellClassifierModel` via [`classifier_model_from_config`](../src/vinson/from_config.py)
- `model_arch.cell_embed` — kwargs for the shared `MLPBlock` trunk: `n_inputs`, `hidden_dims`, `dropout`, `activations`, `batch_norm`, `batch_norm_momentum`, `order`
- `model_kwargs.output_dict` — dict mapping each classification head's name to its number of classes, e.g. `{cell_type: 10, pathological_state: 2}`
- `hparams` — `optimizer_kwargs.lr`, `lr_scheduler`, `lr_scheduler_kwargs`, as for the other models

```yaml
model_type: classifier

model_arch:
  cell_embed:
    n_inputs: 1000
    hidden_dims: [128, 128]
    dropout: 0.2
    activations: silu
    batch_norm: true

model_kwargs:
  output_dict:
    cell_type: 10
    pathological_state: 2
```

# Training data format

Training examples are per-sample motif-proportion embeddings, not sequences. `EncoderDataset` ([`vinson/datasets/cell_classifier.py`](../src/vinson/datasets/cell_classifier.py)) takes:

- `embeddings` — a `pandas.DataFrame` of shape `(n_samples, n_features)`, indexed by `sample_id`. In practice this is derived by normalizing raw motif proportions per sample (log transform + z-score) and, optionally, aggregating replicates/clusters by median before normalizing.
- `labels` — a `pandas.DataFrame` of shape `(n_samples, n_tasks)`, indexed by `sample_id` (must align with `embeddings`), one column per classification head (e.g. `cell_type`, `pathological_state`). Each column is fit with its own `sklearn.preprocessing.LabelEncoder` (pass a shared `encoders` dict to reuse encoders fit on the full label set, e.g. across train/validation splits).
- `noise` — optional std. dev. of Gaussian noise added to the embedding at `__getitem__` time, used during training as a regularizer (set to `0` for validation/eval sets).

# Training

The model trains like any other Lightning module:

```python
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightning as L

from vinson.datasets.cell_classifier import EncoderDataset
from vinson.models.cell_classifier import CellClassifierModel
from vinson.models.shared import MLPBlock

embed_train, embed_valid, labels_train, labels_valid = train_test_split(
    embeddings, labels_df, test_size=0.2, stratify=labels_df["cell_type"], random_state=0,
)
encoders = {col: LabelEncoder().fit(labels_df[col]) for col in labels_df.columns}

ds_train = EncoderDataset(embed_train, labels_train, encoders=encoders, noise=0.01)
ds_valid = EncoderDataset(embed_valid, labels_valid, encoders=encoders)

train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
valid_loader = DataLoader(ds_valid, batch_size=32)

model = CellClassifierModel(
    embedding=MLPBlock(n_inputs=embed_train.shape[1], hidden_dims=[128, 128], dropout=0.2),
    output_dict={col: len(enc.classes_) for col, enc in encoders.items()},
)

trainer = L.Trainer(max_epochs=1000, accelerator="gpu", devices=1)
trainer.fit(model, train_loader, valid_loader)
```
