from vinson.from_config import read_configs, dhs_model_from_config
from vinson.utils.data_formatting import extract_data_from_backed_anndata
from vinson.utils.sequence_utils import one_hot_encode
from vinson.datasets.sequence import SequenceEmbedDataset
from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor
from genome_tools.data.anndata import read_zarr_backed
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch
import numpy as np
import pandas as pd
import sys
from copy import deepcopy


from predict_DHS_model import predict_from_checkpoint


class PredictDataset(Dataset):
    def __init__(self, pred_inputs):
        self.pred_inputs = pred_inputs

    def __len__(self):
        return len(self.pred_inputs)

    def __getitem__(self, idx):
        inp = self.pred_inputs[idx]
        return {
            x: inp[x] for x in ['ohe_seq', 'embed']
        }
    # def __getitems__(self, idcs):
    #     return [
    #         x for x in self.pred_inputs[idcs]
    #     ]



def make_predict_dict(species, seq, anndata, sample_ids):
    return [
        {
            'species': species,
            'seq': seq,
            'ohe_seq': one_hot_encode(seq),
            'sample_id': sample_ids,
            'embed': anndata.obsm['motif_embeddings'].loc[sample_id].values.astype(np.float32)
        }
        for sample_id in sample_ids
    ]

def pad_with_N(seq, num=100):
    res = 'N' * num + seq[num:-num] + 'N' * num
    assert len(res) == len(seq)
    return res


def get_dataloader(predict_data, anndata):
    sample_ids = anndata.obs.index.values
    embeds = anndata.obsm.loc[sample_ids].values.astype(np.float32)
    predict_data['sample_id'] = [sample_ids] * len(predict_data)
    predict_data['embed'] = [embeds] * len(predict_data)
    predict_data = predict_data.explode(['sample_id', 'embed'])
    predict_data['ohe_seq'] = predict_data['seq'].apply(one_hot_encode)
    predict_data = predict_data.to_dict(orient='records')
    pred_ds = PredictDataset(predict_data)
    pred_dl = DataLoader(pred_ds, batch_size=128, num_workers=6, drop_last=False, shuffle=False)

    return pred_dl


if __name__ == "__main__":

    print(sys.argv)
    prefix = sys.argv[1]

    with open(sys.argv[2]) as f:
        fasta = f.readline().strip()

    predict_meta = pd.DataFrame(
        {
            'prefix': [prefix],
            'seq': [fasta]
        }
    )
    anndata = read_zarr_backed(sys.argv[3])

    dl = get_dataloader(predict_meta, anndata)

    model_config = read_configs(sys.argv[5])

    predictions = predict_from_checkpoint(
        model_config,
        model_checkpoint=sys.argv[4],
        dataloader=dl
    )
    np.save(sys.argv[6], predictions)
