import torch
import sys
from .sequence import CellEmbedding, EmbedModel, BassetTrunkEmbed
from .variant import VariantEmbedModel


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


def make_dhs_model(model_arch, **model_kwargs):
    
    # TODO: parse model_arch
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)


    model = EmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        regression=True,
        **model_kwargs,
        **model_kwargs
        
    )
    model.init_model()
    return model


def make_variant_model(model_arch, **model_kwargs):
    # TODO: parse model_arch

    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)


    model = VariantEmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        **model_kwargs,
    )
    model.init_model()
    return model
