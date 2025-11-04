import torch
import lightning as L


class EmbeddingMLP(torch.nn.Module):

    def __init__(self, n_inputs, n_nodes=64, n_layers=1, dropout=0.3):
        """ """
        super(EmbeddingMLP, self).__init__()

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
            [torch.nn.BatchNorm1d(num_features=n_nodes) for i in range(self.n_layers)]
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
    def __init__(self, n_inputs, n_cell_categories, **kwargs):
        super().__init__()
        
        self.trunk = EmbeddingMLP(n_inputs, **kwargs)
        self.head = torch.nn.Linear(self.trunk.n_nodes, n_cell_categories)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.head(self.trunk(x))
    
    def step(self, batch):
        X, y = (
            batch["embed"],
            batch["cell_category"],
        )

        y_hat = self(X)
        loss = self.loss_fn(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-5)
        return optimizer


class CellAndPathologicalStateClassifierModel(L.LightningModule):
    def __init__(self, n_inputs, n_cell_categories, n_pathological_states, **kwargs):
        super(CellAndPathologicalStateClassifierModel, self).__init__()
        
        self.trunk = EmbeddingMLP(n_inputs, **kwargs)
        self.head_cell_category = torch.nn.Linear(self.trunk.n_nodes, n_cell_categories)
        self.head_pathological_state = torch.nn.Linear(self.trunk.n_nodes, n_pathological_states)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.trunk(x)
        cell_category = self.head_cell_category(x)
        pathological_state = self.head_pathological_state(x)
        return cell_category, pathological_state

    def training_step(self, batch, batch_idx):
        X, y_cell_category, y_pathological_state = (
            batch["embed"],
            batch["cell_category"],
            batch["pathological_state"],
        )

        _y_cell_category, _y_pathological_state = self(X)

        loss_cell_category = self.loss_fn(_y_cell_category, y_cell_category)
        loss_pathological_state = self.loss_fn(_y_pathological_state, y_pathological_state)
        loss = loss_cell_category + loss_pathological_state

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X, y_cell_category, y_pathological_state = (
            batch["embed"],
            batch["cell_category"],
            batch["pathological_state"],
        )

        _y_cell_category, _y_pathological_state = self(X)

        loss_cell_categories = self.loss_fn(_y_cell_category, y_cell_category)
        loss_pathological_state = self.loss_fn(_y_pathological_state, y_pathological_state)
        loss = loss_cell_categories + loss_pathological_state

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
        return optimizer
