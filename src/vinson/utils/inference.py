from tqdm import tqdm_notebook, tqdm
from dataclasses import dataclass, field, asdict 

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.utils.data import  DataLoader

from genome_tools.data.extractors import FastaExtractor
from vinson.datasets.sequence import CartesianInferenceDataset
from vinson.datasets.variant import CartesianVariantInferenceDataset
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

import pandas as pd
@dataclass
class InferenceDatasetKwargs:
    embed_df: pd.DataFrame
    fasta_file: str
    bed_df: pd.DataFrame | None = None
    variants_df: pd.DataFrame | None = None
    shifts: ArrayLike = field(default_factory=lambda: np.array([]))
    embed_meta_df: None | pd.DataFrame  = None
    genotype_file: None | str = None

    def __post_init__(self):
        if not (
            (self.bed_df is not None) ^ (self.variants_df is not None)
                ):
            raise ValueError('One of variants_df or bed_df should be used')
        
        if (
            (self.embed_meta_df is not None) ^ (self.genotype_file is not None)
            ):
            raise ValueError('Both embed_meta_df and genotype_file should be passed to inject genotypes')

    def to_dict(self):
        return asdict(self, dict_factory=lambda x: {k:v for k,v in x if v is not None})
    

def score_bed_with_dhs_model(inference_dataclass: InferenceDatasetKwargs,
                             model,
                             dataloader_kwargs =dict(num_workers=1, batch_size=128), 
                             device='cpu'):
    
    dataset = CartesianInferenceDataset(**inference_dataclass.to_dict()) 
    dataloader = DataLoader(dataset, 
                            **dataloader_kwargs)
    
    lst_preds = []
    with torch.no_grad():    
        for batch in tqdm_notebook(dataloader):
            seq, seq_revcomp = batch['seq'].to(device), batch['seq_revcomp'].to(device)
            embeds = batch['embed'].to(device)

            pred = model(seq, embeds)
            pred_revcomp = model(seq_revcomp, embeds)
            preds = (torch.log(pred) + torch.log(pred_revcomp))/2
            preds = torch.exp(preds)

            lst_preds.append(preds.cpu().numpy())

    predicted_scores = np.concatenate(lst_preds)
    out_df = dataset.prepare_meta()
    return out_df, predicted_scores

def score_variants_with_dhs_model(inference_dataclass: InferenceDatasetKwargs,
                                  model,
                                  dataloader_kwargs =dict(num_workers=1, batch_size=128), 
                                  device='cpu'):

    dataset = CartesianVariantInferenceDataset(**inference_dataclass.to_dict()) 
    dataloader = DataLoader(dataset, 
                            **dataloader_kwargs)
    lst_ref, lst_alt = [], []
    model = model.to(device)

    with torch.no_grad():    
        for batch in tqdm_notebook(dataloader):
            ref, ref_revcomp = batch['ref'].to(device), batch['ref_revcomp'].to(device)
            alt, alt_revcomp = batch['alt'].to(device), batch['alt_revcomp'].to(device)
            embeds = batch['embed'].to(device)

            pred_ref, pred_ref_revcomp = model(ref, embeds), model(ref_revcomp, embeds)
            pred_alt, pred_alt_revcomp = model(alt, embeds), model(alt_revcomp, embeds)

            ## geom mean
            pred_ref = (torch.log(pred_ref) + torch.log(pred_ref_revcomp))/2
            pred_alt = (torch.log(pred_alt) + torch.log(pred_alt_revcomp))/2
            pred_ref, pred_alt = torch.exp(pred_ref), torch.exp(pred_alt)
            
            lst_ref.append(pred_ref.cpu().numpy()), lst_alt.append(pred_alt.cpu().numpy())

    ref_scores, alt_scores = np.concatenate(lst_ref), np.concatenate(lst_alt)
    out_df = dataset.prepare_meta()
    # out_df['vinson.ref_score'] = ref_scores
    # out_df['vinson.alt_score'] = alt_scores
    return out_df, (alt_scores, ref_scores)
