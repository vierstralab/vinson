import torch

import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
)
from torchmetrics.regression import PearsonCorrCoef

from vinson.loss import (
    poisson_loss,
    binomial_mixture_loss,
)


class CellEmbedding(torch.nn.Module):
    """
    This a simple multilayer perceptron to encode cell states from an embedding
    'n_inputs' is the dimension of the embedding space.
    """

    def __init__(self, n_inputs, n_nodes=1024, n_outputs=128, n_layers=0):
        super(CellEmbedding, self).__init__()

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.n_layers = n_layers

        self.ifc = torch.nn.Linear(n_inputs, n_nodes)
        self.irelu = torch.nn.ReLU()

        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for i in range(n_layers)]
        )
        self.relus = torch.nn.ModuleList([torch.nn.ReLU() for i in range(n_layers)])

        self.ffc = torch.nn.Linear(n_nodes, n_outputs)

    def forward(self, embed):
        x = self.irelu(self.ifc(embed))
        for i in range(self.n_layers):
            x = self.relus[i](self.fcs[i](x))
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

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        # flatten
        x = self.flatten(x)

        return x


class BassetTrunkEmbed(BassetTrunk):
    def __init__(self, embed):
        super(BassetTrunkEmbed, self).__init__()

        self.embed = embed

        # TODO: inference output size from BassetTrunk convolutional layers
        self.bias2 = torch.nn.Linear(embed.n_outputs, 200)
        self.bias3 = torch.nn.Linear(embed.n_outputs, 200)

    def forward(self, x, embedding):
        embedding = self.embed(embedding)

        x = self.layer1(x)
        x = self.relu1(x)

        x_conv = self.layer2(x)
        x_bias = self.bias2(embedding).unsqueeze(-1)
        x = self.relu2(x_conv + x_bias)

        x_conv = self.layer3(x)
        x_bias = self.bias3(embedding).unsqueeze(-1)
        x = self.relu3(x_conv + x_bias)

        x = self.flatten(x)

        return x


class VinsonModel(L.LightningModule):
    def __init__(self, trunk, seqlen=1344, regression=False, pos_weight=1):
        super(VinsonModel, self).__init__()

        self.trunk = trunk
        self.seqlen = seqlen
        self.regression = regression
        self.pos_weight = pos_weight

        # FC layers
        self.fc1 = torch.nn.LazyLinear(out_features=1024)
        self.bn1 = torch.nn.BatchNorm1d(num_features=1024, momentum=0.1)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.LazyLinear(out_features=1024)
        self.bn2 = torch.nn.BatchNorm1d(num_features=1024, momentum=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.relu2 = torch.nn.ReLU()

        self.final = torch.nn.LazyLinear(out_features=1)

        # init metrics
        self.init_metrics()

        # init model
        self.init_model()

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

    def forward_final(self, x):
        x = self.final(x)

        # return x.squeeze()
        return x

    def forward(self, seq):
        features = self.trunk(seq)

        x = self.forward_fc(features)
        x = self.forward_final(x)

        return x

    def training_step(self, batch, batch_idx):
        X_seq, indicator, density, read_depth, r = (
            batch["seq"],
            batch["indicator"],
            batch["density"],
            batch["read_depth"],
            batch["r"],
        )

        y = self(X_seq).squeeze()

        if self.regression:
            # Transform normalized density to counts
            # The model ouputs the log counts
            pred_counts = (torch.exp(y) / 1e6 * read_depth) + 1.0
            target_counts = (density / 1e6 * read_depth) + 1.0

            loss = poisson_loss(pred_counts, target_counts)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y, indicator.float(), pos_weight=torch.tensor(self.pos_weight)
            )

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X_seq, indicator, density, read_depth, r = (
            batch["seq"],
            batch["indicator"],
            batch["density"],
            batch["read_depth"],
            batch["r"],
        )

        y = self(X_seq).squeeze()

        if self.regression:
            # Transform normalized density to counts
            # The model ouputs the log counts
            pred_counts = (torch.exp(y) / 1e6 * read_depth) + 1.0
            target_counts = (density / 1e6 * read_depth) + 1.0

            loss = poisson_loss(pred_counts, target_counts)

            self.valid_metrics.update(pred_counts.log(), target_counts.log())

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0005)


class VinsonEmbedModel(VinsonModel):
    def __init__(self, *args, **kwargs):
        super(VinsonEmbedModel, self).__init__(*args, **kwargs)

    def init_model(self):
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.trunk.embed.n_inputs)),
        )

    def forward(self, seq, embed):
        features = self.trunk(seq, embed)

        x = self.forward_fc(features)
        x = self.forward_final(x)

        return x

    def training_step(self, batch, batch_idx):
        X_seq, X_embed, indicator, density, read_depth, r = (
            batch["seq"],
            batch["embed"],
            batch["indicator"],
            batch["density"],
            batch["read_depth"],
            batch["r"],
        )

        y = self(X_seq, X_embed).squeeze()

        if self.regression:
            # Transform normalized density to counts
            # The model ouputs the log counts
            pred_counts = (torch.exp(y) / 1e6 * read_depth) + 1.0
            target_counts = (density / 1e6 * read_depth) + 1.0

            loss = poisson_loss(pred_counts, target_counts)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y, indicator.float(), pos_weight=torch.tensor(self.pos_weight)
            )

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        X_seq, X_embed, indicator, density, read_depth, r = (
            batch["seq"],
            batch["embed"],
            batch["indicator"],
            batch["density"],
            batch["read_depth"],
            batch["r"],
        )

        y = self(X_seq, X_embed).squeeze()

        if self.regression:
            # Transform normalized density to counts
            # The model ouputs the log counts
            pred_counts = (torch.exp(y) / 1e6 * read_depth) + 1.0
            target_counts = (density / 1e6 * read_depth) + 1.0

            loss = poisson_loss(pred_counts, target_counts)

            self.valid_metrics.update(pred_counts.log(), target_counts.log())

        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                y, indicator.float(), pos_weight=torch.tensor(self.pos_weight)
            )
            self.valid_metrics.update(torch.sigmoid(y), indicator.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss


class VinsonVariantEmbedModel(VinsonModel):
    def __init__(self, *args, **kwargs):
        super(VinsonVariantEmbedModel, self).__init__(*args, **kwargs)

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
            torch.zeros((2, self.trunk.embed.n_inputs)),
        )

    def forward(self, seq_ref, seq_alt, embed):
        ref_features = self.trunk(seq_ref, embed)
        alt_features = self.trunk(seq_alt, embed)
        features = ref_features - alt_features

        x = self.forward_fc(features)
        x = self.forward_final(x)

        return x

    def training_step(self, batch, batch_idx):
        X_seq_ref, X_seq_alt, X_embed, ref_counts, total_counts, bad_score = (
            batch["seq_ref"],
            batch["seq_alt"],
            batch["embed"],
            batch["ref_counts"],
            batch["total_counts"],
            batch["bad_score"],
        )

        y = self(X_seq_ref, X_seq_alt, X_embed).squeeze()

        loss = binomial_mixture_loss(y, ref_counts, total_counts, bad_score)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X_seq_ref, X_seq_alt, X_embed, ref_counts, total_counts, bad_score = (
            batch["seq_ref"],
            batch["seq_alt"],
            batch["embed"],
            batch["ref_counts"],
            batch["total_counts"],
            batch["bad_score"],
        )

        y = self(X_seq_ref, X_seq_alt, X_embed).squeeze()

        loss = binomial_mixture_loss(y, ref_counts, total_counts, bad_score)

        self.valid_metrics.update(
            y, torch.log(ref_counts / (total_counts - ref_counts))
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
