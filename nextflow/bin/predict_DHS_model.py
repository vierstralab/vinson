import torch
import sys
import numpy as np


from torch.utils.data import DataLoader


sys.path.append('/home/jvierstra/proj/vinson')

from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.models.sequence import BassetTrunkEmbed, CellEmbedding, EmbedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm
import argparse


def load_vinson_model(checkpoint_path, device):
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    model_predict = EmbedModel(
        trunk_model,
        embed_model,
        regression=True,
    ).to(device)
    pretrained_state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )["state_dict"]

    model_predict.load_state_dict(pretrained_state_dict)
    model_predict = model_predict.eval()
    return model_predict


def load_legnet_model(checkpoint_path, device):
    sys.path.append('/home/sabramov/packages/dnase_legnet')
    from dnase_legnet.legnet_embed_cnn import LegNetEmbedinCNN

    model = LegNetEmbedinCNN.load_from_checkpoint(checkpoint_path, map_location=device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict DHS model")
    parser.add_argument("dhs_dataset", type=str, help="Path to DHS dataset (.h5 file)")
    parser.add_argument("--embeddings_file", type=str, default="/home/jvierstra/proj/vinson/data/embeddings_old_clustername.tsv")
    parser.add_argument("--fasta_file", type=str, default="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa")
    parser.add_argument("--sample_genotype_file", type=str, default="/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/meta+sample_ids.tsv")
    parser.add_argument("--genotype_file", type=str, default="/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/all_variants_stats.bed.gz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_v2/checkpoints/epoch=3-step=1851994-val_loss=16.16.ckpt")
    parser.add_argument("--model_type", type=str, default="vinson", choices=["vinson", "legnet"], help="Type of model to use for prediction")
    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()

    dataset_kwargs = dict(
        sample_genotype_file=args.sample_genotype_file,
        genotype_file=args.genotype_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
    )

    dataset = SequenceEmbedDataset(
        args.dhs_dataset,
        args.embeddings_file,
        args.fasta_file,
        negative_samples_rate=0,
        **dataset_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    if args.model_type == "vinson":
        model_predict = load_vinson_model(args.checkpoint, device)
    elif args.model_type == "legnet":
        model_predict = load_legnet_model(args.checkpoint, device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    @torch.inference_mode()
    def load_and_predict(batch, model):    
        X_seq      = batch["ohe_seq"].to(device, non_blocking=True)
        X_embed    = batch["embed"].to(device, non_blocking=True)
        y_ = model(X_seq, X_embed).squeeze().detach().cpu().numpy()
        return y_

    y_hat_all = np.concatenate([load_and_predict(batch, model_predict) for batch in tqdm(dataloader)])
    np.save(args.output, y_hat_all)

if __name__ == "__main__":
    main()