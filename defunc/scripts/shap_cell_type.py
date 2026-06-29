import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
torch.cuda.empty_cache()
from tangermeme.predict import predict
from tangermeme.product import apply_pairwise
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear
from vinson.interpretation import dinucleotide_shuffle, force_strict_ohe
from vinson.vinson.utils.sequence_utils import intervals_to_ohe
from vinson.models.helpers import _Exp
from vinson.models.sequence import BassetTrunkEmbed, CellEmbedding, EmbedModel
from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor

def main(bed_file, fasta_file, embeddings_file, output_file,
         ag, model_file, seqlen=1344, widen=672):

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk = BassetTrunkEmbed(embed.n_outputs)
    model = EmbedModel(trunk, embed, regression=True)
    pretrained_state_dict = torch.load(
    model_file,
    map_location=torch.device("cpu"),)["state_dict"]
    model.load_state_dict(pretrained_state_dict)
    model = model.to(device)
    model = model.eval()
    
    # Load fasta extractor
    fasta_extr = FastaExtractor(fasta_file)
    bed_df = pd.read_csv(bed_file, sep="\t", usecols=[0, 1, 2])

    # Make list of widened intervals
    intervals = []
    for _, row in bed_df.iterrows():
        mid = ((int(row["start"]) + int(row["end"])) // 2)
        interval = GenomicInterval(row["#chr"], mid, mid).widen(672)
        intervals.append(interval)
        
     
    embeddings_df = pd.read_csv(embeddings_file, sep="\t", index_col=0)
    filtered_embedding = embeddings_df[ag].values
    
    X_all = intervals_to_ohe(intervals, seqlen=1344, fasta_extr=fasta_extr)
    _X = torch.tensor(X_all, dtype=torch.float32, device="cpu")  # stay on CPU
    X_embed_single = torch.tensor(filtered_embedding, dtype=torch.float32, device="cpu").unsqueeze(0)
      
    # Store all SHAP results
    all_attrs = []
    
    #torch.cuda.empty_cache()
    #trying to run in batch
    batch_size = 32
    y_embed = X_embed_single.repeat(_X.shape[0], 1)  # (num_intervals, embedding_dim)
    
    for i in tqdm(range(0, _X.shape[0], batch_size), desc="Running DeepLIFT/SHAP"):
        _x = force_strict_ohe(_X[i:i+batch_size]).to(device)  # move batch only
        _y = X_embed_single.repeat(len(_x), 1).to(device)

        attrs = deep_lift_shap(
            model,
            _x,
            args=(_y,),
            device=device,
            print_convergence_deltas=False,
            references=dinucleotide_shuffle,
            additional_nonlinear_ops={_Exp: _nonlinear}
        )

        all_attrs.extend([a.cpu() for a in attrs])  # move results back to CPU
        del _x, _y, attrs
        torch.cuda.empty_cache()
        
    # Stack into tensor
    attrs_stack = torch.stack(all_attrs, dim=0)

    # Attribution × input
    real_scores = (_X * attrs_stack).sum(axis=1).numpy() 

    # Save to .npy
    np.save(f'{output_file}_real_score.npy', real_scores)
    np.save(f'{output_file}_attr.npy', attrs_stack)
    print(f"[done] Saved attribution matrix to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepLIFT/SHAP on intervals")
    parser.add_argument("--bed", required=True, help="BED file with peaks")
    parser.add_argument("--fasta", required=True, help="Reference FASTA file")
    parser.add_argument("--embeddings", required=True, help="Embeddings TSV")
    parser.add_argument("--model-file", required=True, help="model checkpoint file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--ag", default="T-cell", help="ag to choose cell type")
    args = parser.parse_args()

    main(
        bed_file=args.bed,
        fasta_file=args.fasta,
        model_file=args.model_file,
        embeddings_file=args.embeddings,
        output_file=args.output,
        ag=args.ag
    )