import torch
import copy

import lightning as L

import csv
import os

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
)
from torchmetrics.regression import PearsonCorrCoef
from torch.nn import BCEWithLogitsLoss

from vinson.loss import (
    PoissonNLL,
    binomial_mixture_normed_loss,
)
from vinson.lr import CosineAnnealingWarmupRestarts
from vinson.models.cell_classifier import EmbeddingMLP


class CellEmbedding(EmbeddingMLP):
    """
    This a simple multilayer perceptron to encode cell states from an embedding
    'n_inputs' is the dimension of the embedding space.
    """

    def __init__(self, n_inputs, n_nodes=1024, n_outputs=128, n_layers=0):
        super().__init__(n_inputs=n_inputs, n_nodes=n_nodes, n_layers=n_layers)
        self.n_outputs = n_outputs

        self.ffc = torch.nn.Linear(n_nodes, self.n_outputs)

    def forward(self, embed):
        x = super().forward(embed)
        x = self.ffc(x)
        return x


class BassetTrunk(torch.nn.Module):
    def __init__(self):
        super(BassetTrunk, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=4, out_channels=300, kernel_size=19, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=300, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, padding=(3 - 1) // 2),
        )
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=300, out_channels=200, kernel_size=11, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=200, out_channels=200, kernel_size=7, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu3 = torch.nn.ReLU()

        # self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        # flatten
        # x = self.flatten(x)
        x = torch.flatten(x, start_dim=1)

        return x


class BassetTrunkEmbed(BassetTrunk):
    def __init__(self, n_embed_outputs):
        super().__init__()

        # TODO: inference output size from BassetTrunk convolutional layers
        self.bias2 = torch.nn.Linear(n_embed_outputs, self.layer2[0].out_channels)
        self.bias3 = torch.nn.Linear(n_embed_outputs, self.layer3[0].out_channels)

    def forward(self, x, embed):
        x = self.layer1(x)
        x = self.relu1(x)

        x_conv = self.layer2(x)
        x_bias = self.bias2(embed).unsqueeze(-1)
        x = self.relu2(x_conv + x_bias)

        x_conv = self.layer3(x)
        x_bias = self.bias3(embed).unsqueeze(-1)
        x = self.relu3(x_conv + x_bias)

        # Flatten features
        x = torch.flatten(x, start_dim=1)

        return x


class AbstractBaseSequenceModel(L.LightningModule):
    def __init__(
        self,
        trunk_model,
        seqlen=1344,
        lr_scheduler=None,
        optimizer_kwargs=dict(),
        lr_scheduler_kwargs=dict(),
    ):
        super().__init__()

        self.trunk = trunk_model
        self.seqlen = seqlen

        # Common architecture
        self.fc1 = torch.nn.LazyLinear(1024)
        self.bn1 = torch.nn.BatchNorm1d(1024, momentum=0.1)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.LazyLinear(1024)
        self.bn2 = torch.nn.BatchNorm1d(1024, momentum=0.1)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.relu2 = torch.nn.ReLU()

        self.final = torch.nn.LazyLinear(1)

        # Optimizer setup
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.train_metrics = MetricCollection({}, prefix="train_")
        self.valid_metrics = MetricCollection({}, prefix="val_")

    def init_metrics(self):
        raise NotImplementedError("Subclasses of AbstractBaseSequenceModel must implement init_metrics method.")

    def forward_fc(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        return x

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        """ """
        lr_scheduler_dict = {
            "CosineAnnealingWarmupRestarts": CosineAnnealingWarmupRestarts,
            "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR
        }
        lr_scheduler = lr_scheduler_dict.get(self.lr_scheduler, None) 
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)

        if lr_scheduler is None:
            return optimizer

        scheduler = lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }


class BaseSequenceModel(AbstractBaseSequenceModel):
    def __init__(
        self,
        trunk_model,
        seqlen=1344,
        regression=False,
        log_output=True,
        lr_scheduler: str=None,
        optimizer_kwargs=dict(),
        lr_scheduler_kwargs=dict(),
    ):
        super().__init__(
            trunk_model=trunk_model,
            seqlen=seqlen,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.regression = regression
        self.log_output = log_output

        self.loss = (
            PoissonNLL(log_input=log_output, reduction="none")
            if self.regression
            else BCEWithLogitsLoss(reduction="none")
        )
        
        
        self.init_metrics()


    def init_metrics(self):
        if self.regression:
            self.train_metrics = MetricCollection(
                {
                    "pcc": PearsonCorrCoef(),
                },
                prefix="train_",
            )
        else:
            self.train_metrics = MetricCollection(
                {
                    "auroc": BinaryAUROC(),
                    "aupr": BinaryAveragePrecision(),
                    "mcc": BinaryMatthewsCorrCoef(),
                },
                prefix="train_",
            )

        self.valid_metrics = self.train_metrics.clone(prefix="val_")

    def init_model(self):
        self(torch.zeros((2, 4, self.seqlen)))
        return self

    def forward(self, seq):
        features = self.trunk(seq)

        x = self.forward_fc(features)
        x = self.forward_final(x)
   
        return x
    
    def forward_final(self, x):
        x = self.final(x)
        if not self.log_output:
            x = torch.relu(x)
        return x

    def _forward_from_batch(self, batch):
        X_seq = batch["ohe_seq"]
        y = self(X_seq).squeeze()
        return y

    def _run_step_regression(self, y, read_depth, bg, density):
        million = torch.tensor(1e6, device=self.device)

        if self.log_output:
            input = torch.logaddexp(
                y - million.log() + torch.log(read_depth), torch.log(bg)
            )
        else:
            input = y / million * read_depth + bg

        target_counts = torch.clip(density / million * read_depth, bg, None)

        return input, target_counts # y_hat, y

    def _run_step_classification(self, y, indicator):
        return y, indicator.float()  # y_hat, y

    def _run_step(self, batch):
        """
        Internal step function to parse batch and run forward + step

        Returns:
        y: model predictions
        target: ground truth values
        """
        weight = batch["weight"]

        y = self._forward_from_batch(batch)
        if self.regression:
            return self._run_step_regression(
                y,
                read_depth=batch["read_depth"],
                bg=batch["bg"],
                density=batch["density"]
            ), weight
        else:
            return self._run_step_classification(y, batch["class"] == 1), weight

    def step(self, batch, batch_idx):
        (y_hat, y), weight = self._run_step(batch)

        loss = self.loss(y_hat, y)
        loss *= weight
        loss = loss.mean()

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)

        if self.regression:
            if not self.log_output:
                y_hat = torch.log(y_hat + 1e-6)
            self.valid_metrics.update(
                y_hat,
                y
            )
        else:
            self.valid_metrics.update(torch.sigmoid(y_hat), y.int())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()


class EmbedModel(BaseSequenceModel):
    """Sequence with embeddings model"""
    def __init__(self, trunk: BassetTrunkEmbed, embed: CellEmbedding, *args, **kwargs):
        super().__init__(trunk, *args, **kwargs)

        self.embedding = embed

        self.save_hyperparameters(ignore=["trunk", "embed"])

    def init_model(self):
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.embedding.n_inputs)),
        )
        return self
    
    def forward(self, seq, embed):
        x = self.embedding(embed)
        x = self.trunk(seq, x)

        x = self.forward_fc(x)
        x = self.forward_final(x) 

        return x

    def _forward_from_batch(self, batch):
        X_seq = batch["ohe_seq"]
        X_embed = batch["embed"]
        y = self(X_seq, X_embed).squeeze()
        return y

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)


class VariantEmbedModel(AbstractBaseSequenceModel):
    def __init__(self, trunk: BassetTrunkEmbed, embed: CellEmbedding, *args, **kwargs):
        super().__init__(trunk, *args, **kwargs)
    
        self.loss = binomial_mixture_normed_loss
        self.embedding = embed
        self.save_hyperparameters(ignore=["trunk", "embed"])

    def init_metrics(self):
        self.train_metrics = MetricCollection(
            {
                "pcc": PearsonCorrCoef(),
            },
            prefix="train_",
        )

        self.valid_metrics = self.train_metrics.clone(prefix="val_")

    def init_model(self):
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.embedding.n_inputs)),
        )
        
    def forward_final(self, x):
        x = self.final(x)
        return x

    def forward(self, seq_ref, seq_alt, embed):
        x = self.embedding(embed)
        ref_features = self.trunk(seq_ref, x)
        alt_features = self.trunk(seq_alt, x)

        x = torch.subtract(ref_features, alt_features)

        x = self.forward_fc(x)
        x = self.forward_final(x) # in variant model, outputs are always logits of ES -infinity to +infinity
        return x
    
    def _forward_from_batch(self, batch):
        X_ref = batch["ohe_seq_ref"]
        X_alt = batch["ohe_seq_alt"]
        X_embed = batch["embed"]
        y = self(X_ref, X_alt, X_embed).squeeze()
        return y
    
    def step(self, batch, batch_idx):
        y = self._forward_from_batch(batch)

        ref_counts, total_counts, bad_score = (
            batch["ref_counts"],
            batch["total_counts"],
            batch["bad_score"],
        )

        #has predict (y), ref_counts (target), total_counts (n), bad_score (bad_score), optional: tau, reduction
        
        loss = self.loss(
            y, 
            ref_counts,
            total_counts,
            bad_score,
            reduction="none"
        )
        
        # multiply by weights
        loss *= batch["weight"]

        # --- Detect NaN/Inf and print detailed info ---
        nan_indices = torch.where(torch.isnan(loss) | torch.isinf(loss))[0]
        if len(nan_indices) > 0:
            print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}")
            print("Indices with NaN/Inf:", nan_indices.tolist())
    
            for idx in nan_indices:
                idx = idx.item()  # make Python int
                print(f"\nVariant at batch idx {idx}:")
                print(f"  y = {y[idx].item()}")
                print(f"  ref_counts = {ref_counts[idx].item()}")
                print(f"  total_counts = {total_counts[idx].item()}")
                print(f"  bad_score = {bad_score[idx].item()}")
                print(f"  weight = {batch['weight'][idx].item()}")
    
                # print variant metadata if present
                for key in ["chrom", "pos", "ref", "alt", "gt"]:
                    if key in batch:
                        val = batch[key][idx]
                        # convert single-element tensor to Python scalar
                        if torch.is_tensor(val) and val.numel() == 1:
                            val = val.item()
                        print(f"  {key} = {val}")
    
            # Optionally: replace NaN/Inf with large number to continue training
            #loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
            raise RuntimeError(
                f"NaN/Inf loss encountered at batch {batch_idx}. "
                "See diagnostic output above."
            )
    
        loss = loss.mean()


        return loss, y, (ref_counts, total_counts, bad_score)

    def debug_training_step(self, batch, batch_idx):
        # Print batch info
        print(f"\n--- Batch {batch_idx} ---")
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"{k}: shape={v.shape}, min={v.min().item()}, max={v.max().item()}, NaN={torch.isnan(v).any().item()}, Inf={torch.isinf(v).any().item()}")
            else:
                print(f"{k}: type={type(v)}, len={len(v)}")
        
        # Call original step (optional, to see loss)
        loss, *_ = self.step(batch, batch_idx)
        print(f"Batch loss: {loss.item()}")
        return loss

    def debug_validation_step(self, batch, batch_idx):
        print(f"\n--- Validation Batch {batch_idx} ---")
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"{k}: shape={v.shape}, min={v.min().item()}, max={v.max().item()}, NaN={torch.isnan(v).any().item()}, Inf={torch.isinf(v).any().item()}")
            else:
                print(f"{k}: type={type(v)}, len={len(v)}")
        
        loss, y_hat, y = self.step(batch, batch_idx)
        print(f"Validation batch loss: {loss.item()}")
        return loss


    # def training_step(self, batch, batch_idx):
    #     loss, *_ = self.step(batch, batch_idx)

    #     self.log(
    #         "loss", loss, on_step=True, on_epoch=False, sync_dist=True
    #     )

    #     return loss
    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)
    
        if getattr(self, "debug", False) and hasattr(self, "batch_log_file"):
            lfc = batch.get("lfc")
            ref_counts = batch.get("ref_counts")
            total_counts = batch.get("total_counts")
            bad_score = batch.get("bad_score")
    
            with open(self.batch_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    batch_idx,
                    loss.item(),
                    lfc.min().item() if lfc is not None else "",
                    lfc.max().item() if lfc is not None else "",
                    ref_counts.min().item() if ref_counts is not None else "",
                    ref_counts.max().item() if ref_counts is not None else "",
                    total_counts.min().item() if total_counts is not None else "",
                    total_counts.max().item() if total_counts is not None else "",
                    bad_score.min().item() if bad_score is not None else "",
                    bad_score.max().item() if bad_score is not None else "",
                ])
    
        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        lfc = batch.get("lfc")
        self.valid_metrics.update(y_hat, lfc)
    
        if getattr(self, "debug", False) and hasattr(self, "batch_log_file"):
            ref_counts = batch.get("ref_counts")
            total_counts = batch.get("total_counts")
            bad_score = batch.get("bad_score")
    
            with open(self.batch_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    f"val_{batch_idx}",
                    loss.item(),
                    lfc.min().item() if lfc is not None else "",
                    lfc.max().item() if lfc is not None else "",
                    ref_counts.min().item() if ref_counts is not None else "",
                    ref_counts.max().item() if ref_counts is not None else "",
                    total_counts.min().item() if total_counts is not None else "",
                    total_counts.max().item() if total_counts is not None else "",
                    bad_score.min().item() if bad_score is not None else "",
                    bad_score.max().item() if bad_score is not None else "",
                ])
    
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     loss, y_hat, y = self.step(batch, batch_idx)
    #     lfc = batch["lfc"]

    #     self.valid_metrics.update(y_hat, lfc)

    #     self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    #     return loss
   

class VariantEmbedModelWrapper(L.LightningModule):
    """Wrapper class for VariantModel to perform only inference"""

    def __init__(self, model: "VariantEmbedModel"):
        super().__init__()
        self.model = model

        # Make independent ref/alt branches
        self.embedding_ref = copy.deepcopy(model.embedding)
        self.embedding_alt = copy.deepcopy(model.embedding)

        self.trunk_ref = copy.deepcopy(model.trunk)
        self.trunk_alt = copy.deepcopy(model.trunk)
        
        for mod in [
            self.trunk_ref, self.trunk_alt, 
            self.embedding_ref, self.embedding_alt
        ]:
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False
    
    def __getattr__(self, name):
        if name != "model":
            try:
                return getattr(self.model, name)
            except AttributeError:
                pass
        raise AttributeError(f"{self.model} has no attribute {name}")

    def forward(self, seq_ref, seq_alt, embed):
        """ """
        features_ref = self.trunk_ref(seq_ref, self.embedding_ref(embed))
        features_alt = self.trunk_alt(seq_alt, self.embedding_alt(embed.clone()))

        x = torch.subtract(features_ref, features_alt)

        x = self.model.forward_fc(x)
        x = self.model.forward_final(x)

        return x
