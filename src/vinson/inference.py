from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import  DataLoader

import pandas as pd

from vinson.datasets.sequence import InferenceSequenceEmbedDataset
from vinson.utils.data_formatting import VinsonData


def assemble_batch(sample_id, interval, embeddings, fasta, genotype_file=None, step=None, seqlen=1344):
    """
    Assemble a single batch of model inputs for one sample.

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
    genotype_file: str, optional
        VCF to inject the sample's genotypes into the sequence.
    step: int, optional
        If None, one model input centered on the interval center.
        Otherwise, model inputs centered every `step` bp from `interval.start` to `interval.end`
        (e.g. to predict a signal track across the interval).
    seqlen: int, default 1344
        Length of each model input.

    Returns
    -------
    dict
        Batch with 'ohe_seq', 'embed', 'chrom', 'summit', 'sample_id', 'shift', 'revcomp'.
    """
    if step is None:
        summits = np.array([interval.center.start])
    else:
        summits = np.arange(interval.start, interval.end + step, step)

    data = VinsonData.inference_from_bed(
        pd.DataFrame({
            'chrom': interval.chrom,
            'start': summits,
            'end': summits + 1,
            'summit': summits,
        }),
        sample_ids=[sample_id],
        embeddings_df=embeddings.loc[[sample_id], :],
    )
    return assemble_batch_from_data(data, fasta, genotype_file=genotype_file, seqlen=seqlen)


def assemble_batch_from_data(data: VinsonData, fasta, genotype_file=None, seqlen=1344):
    """
    Assemble all examples of an inference `VinsonData` (e.g. from `VinsonData.inference_from_bed`)
    into a single batch.
    """
    dataset = InferenceSequenceEmbedDataset(
        data=data,
        fasta_file=fasta,
        genotype_file=genotype_file,
        seqlen=seqlen,
    )
    dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(dl))


def score_bed_with_dhs_model(
        dataloader,
        model,
        device='cpu'
    ):
    """
    Run a DHS model (or its trunk, to extract embeddings) over every batch of a dataloader built from
    `InferenceSequenceEmbedDataset`, returning the concatenated predictions in dataloader order.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Yields batches with 'ohe_seq' (one-hot sequence) and 'embed' (cell-type embedding).
    model : torch.nn.Module
        Model (or `model.trunk_model` to get pre-aggregated features instead of a single prediction)
        called as `model(ohe_seq, embed)`.
    device : str, default 'cpu'
        Device to move the model and batches to for inference.

    Returns
    -------
    np.ndarray
        Predictions (or features, if `model.trunk_model` was passed) for every example, concatenated
        across batches.
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


def score_variants_with_dhs_model(
        dataloader,
        model,
        device='cpu'
    ):
    """
    Run a DHS model over every batch of a dataloader built from `InferenceVariantEmbedDataset`,
    scoring both the reference and alternate allele sequence for every variant.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Yields batches with 'ohe_seq_ref', 'ohe_seq_alt' (one-hot ref/alt sequences) and 'embed'
        (cell-type embedding).
    model : torch.nn.Module
        Model called as `model(ohe_seq, embed)` for each allele.
    device : str, default 'cpu'
        Device to move the model and batches to for inference.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(alt_scores, ref_scores)`, each concatenated across batches in dataloader order.
    """
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
    return ref_scores, alt_scores
