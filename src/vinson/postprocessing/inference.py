from tqdm import tqdm_notebook, tqdm

import numpy as np
import torch
from torch.utils.data import  DataLoader

from genome_tools.data.extractors import FastaExtractor
from vinson.utils.data_formatting import VinsonData
from vinson.datasets.sequence import InferenceSequenceEmbedDataset

from vinson.utils.sequence_utils import one_hot_encode



def assemble_batch(sample_id, interval, embeddings, fasta):
    """
    Assemble batch from sample_id, interval and embedding
    
    Parameters
    -------------
    sample_id: str
        Sample id, for example AGXXXXX
    interval: GenomicInterval 
        The target genomic window.
    embeddings: pandas.DataFrame
        DataFrame with embeddings, index must be sample_id
    fasta: str
        Path to fasta file with genome 
    """

    with FastaExtractor(fasta) as ext:
        seq = ext[interval]
    dl = DataLoader(
        [
            {
                'ohe_seq': one_hot_encode(seq.upper(), dtype=np.float32),
                'sample_id': sample_id,
                'summit': interval.center.start,
                'chrom': interval.chrom,
                'embed': embeddings.loc[sample_id].values.astype(np.float32)
            }
        ]
    )

    return next(iter(dl))


def score_bed_with_dhs_model(
        dataloader,
        model,
        device='cpu'
    ):
    """
    
    """
    model = model.to(device).eval()

    lst_preds = []
    with torch.no_grad():    
        for batch in tqdm(dataloader):
            seq = batch['ohe_seq'].to(device)
            embeds = batch['embed'].to(device)

            preds = model(seq, embeds)
            lst_preds.append(preds.cpu().numpy())

    predicted_scores = np.concatenate(lst_preds)
    return predicted_scores

def score_variants_with_dhs_model(dataloader,
                                  model,
                                  device='cpu'):

    lst_ref, lst_alt = [], []
    model = model.to(device).eval()

    with torch.no_grad():    
        for batch in tqdm(dataloader):
            ref = batch['ohe_seq_ref'].to(device)
            alt = batch['ohe_seq_alt'].to(device)
            embeds = batch['embed'].to(device)

            pred_ref = model(ref, embeds)
            pred_alt = model(alt, embeds)

            lst_ref.append(pred_ref.cpu().numpy()), lst_alt.append(pred_alt.cpu().numpy())

    ref_scores, alt_scores = np.concatenate(lst_ref), np.concatenate(lst_alt)
    return alt_scores, ref_scores