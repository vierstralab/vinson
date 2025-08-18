import torch

import lightning as L


class EmbeddingMLP(torch.nn.Module):
    def __init__(self, n_inputs, n_nodes=64, n_layers=1, dropout=0.3):
        """ """
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_layers = n_layers

        self.ifc = torch.nn.Linear(n_inputs, n_nodes)
        self.ibn = torch.nn.BatchNorm1d(num_features=n_nodes)
        self.irelu = torch.nn.ReLU()
        self.idropout = torch.nn.Dropout(p=dropout)

        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for i in range(self.n_layers)]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm(num_features=n_nodes) for i in range(self.n_layers)]
        )
        self.relus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(self.n_layers)]
        )
        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for i in range(self.n_layers)]
        )

    def forward(self, x):
        """ """
        x = self.ifc(x)
        x = self.ibn(x)
        x = self.irelu(x)
        x = self.idropout(x)

        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.bns[i](x)
            x = self.relus[i](x)
            x = self.dropouts[i](x)

        return x


class CellClassifierModel(L.LightningModule):
    def __init__(self, n_inputs, n_cell_types, **kwargs):
        self.trunk = EmbeddingMLP(n_inputs, **kwargs)
        self.head = torch.nn.Linear(self.trunk.n_nodes, n_cell_types)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.head(self.trunk(x))

    def training_step(self, batch, batch_idx):
        X, y = batch

        y_ = self(X)
        loss = self.loss_fn(y_, y)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        _y = self(X)
        loss = self.loss_fn(_y, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)
        return optimizer

class CellAndDiseaseStateClassifierModel(L.LightningModule):
    def __init__(self, n_inputs, n_cell_types, n_disease_states, **kwargs):
        self.trunk = EmbeddingMLP(n_inputs, **kwargs)
        self.head_cell_type = torch.nn.Linear(self.trunk.n_nodes, n_cell_types)
        self.head_disease_state = torch.nn.Linear(
            self.trunk.n_nodes, n_disease_states
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.trunk(x)
        cell_type = self.head_cell_type(x)
        disease_state = self.head_disease_state(x)
        return cell_type, disease_state

    def train_step(self, batch, batch_idx):
        X, y_cell_type, y_disease_state = batch

        _y_cell_type, _y_disease_state = self(X)

        loss_cell_type = self.loss_fn(_y_cell_type, y_cell_type)
        loss_disease_state = self.loss_fn(_y_disease_state, y_disease_state)
        loss = loss_cell_type + loss_disease_state

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, y_cell_type, y_disease_state = batch

        _y_cell_type, _y_disease_state = self(X)

        loss_cell_type = self.loss_fn(_y_cell_type, y_cell_type)
        loss_disease_state = self.loss_fn(_y_disease_state, y_disease_state)
        loss = loss_cell_type + loss_disease_state

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)
        return optimizer