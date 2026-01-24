import sys
from .sequence import EmbedModel, BassetTrunkEmbed
from .variant import VariantEmbedModel
from .cell_classifier import CellEmbedding, EmbeddingMLP


try:
    from dnase_legnet.legnet_embed_cnn import LegNetEmbedInCNN
except ImportError:
    print("Please install dnase_legnet to use LegNet models.", file=sys.stderr)
    sys.exit(1)


def make_legnet_model(config):
    return LegNetEmbedInCNN(
        model_kws=config["model_arch"],
        hparams=config['hparams'],
        # hparams={
        #     "lr_scheduler": scheduler_name,
        #     "lr_scheduler_kwargs": scheduler_kwargs,
        #     "optimizer_kwargs": optimizer_kwargs,
        # },
        **config["model_kwargs"],
    )


def make_dhs_model(model_arch, lr_scheduler, lr_scheduler_kwargs, optimizer_kwargs, **model_kwargs):

    mlp_embedding = EmbeddingMLP(**model_arch["cell_embedding"])
    embed_model = CellEmbedding(mlp_embedding, n_outputs=model_arch['n_outputs'])
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    model = EmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        regression=True,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        **model_kwargs 
    )

    model.init_model()
    return model


def make_variant_model(model_arch, lr_scheduler, lr_scheduler_kwargs, optimizer_kwargs, **model_kwargs):
    mlp_embedding = EmbeddingMLP(**model_arch["cell_embedding"])
    embed_model = CellEmbedding(mlp_embedding, n_outputs=model_arch['n_outputs'])
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    model = VariantEmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        **model_kwargs,
    )

    model.init_model()
    return model
