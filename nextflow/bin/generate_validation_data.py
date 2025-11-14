import numpy as np
import pandas as pd
import argparse
import anndata as ad

from genome_tools.data.anndata import read_zarr_backed
from genome_tools.data.extractors import TabixExtractor
from genome_tools import df_to_genomic_intervals

from vinson.utils.data_formatting import extract_data_from_backed_anndata as extract_dhs_data_from_backed_anndata
from vinson.utils.data_formatting import data_to_h5, sanitize_data


def get_bg_for_peaks(peaks_df, stats_path):
    intervals = df_to_genomic_intervals(peaks_df)
    rows = []
    with TabixExtractor(stats_path) as extractor:
        for interval in intervals:
            df_slice = extractor[interval].query('fit_type == "segment')
            assert len(df_slice) == 1, "Expected exactly one matching stats row per peak"
            rows.append(df_slice)
    stats = pd.concat(rows)
    bg = stats.eval('bg_r * bg_p / (1 - bg_p)').values
    return bg


def generate_data_from_sample_peaks(anndata: ad.AnnData, sample_ids) -> dict:
    anndata_slice = anndata[sample_ids, :]
    data = []
    for sample_id, row in anndata_slice.obs.iterrows():
        sample_slice = anndata_slice[sample_id, :]
        fit_stats_file = row['hotspot3_fit_stats_file']
        peaks = pd.read_table(row['peaks_file_0.01fdr']).drop(
            columns=['start']
        ).rename(
            columns={'summit': 'start'}
        )
        
        peaks['end'] = peaks['start'] + 1
        if 'indiv_id' not in sample_slice.obsm:
            indiv_id = indiv_map = pd.read_table(
                "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/metadata.clustered.tsv"
            ).set_index("sample_id").loc[sample_id, "indiv_id"]
        else:
            indiv_id = sample_slice.obsm['indiv_id']
        data_bundle = {
            'dhs_id': f'{sample_id}.' + peaks.index.astype(str).values,
            'sample_id': np.full(len(peaks), sample_id, dtype=np.str_),
            'read_depth': np.full(len(peaks), row['nuclear_reads'], dtype=np.float32),
            'chrom': peaks['#chr'].values,
            'summit': peaks['start'].values,
            'background': get_bg_for_peaks(peaks, fit_stats_file),
            'class': np.ones(len(peaks), dtype=np.int8),
            'density': peaks['summit_density'].values,
            'indiv_id': np.full(len(peaks), indiv_id, dtype=np.str_),
        }
        data.append(data_bundle)
    
    data = {k: np.concatenate([d[k] for d in data]) for k in data[0].keys()}

    return sanitize_data(data)

def check_none(val):
    if val is None or val == 'None' or pd.isna(val):
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate validation data")
    parser.add_argument("anndata_file", type=str, help="Input AnnData file.")
    parser.add_argument("output_file", type=str, help="Output HDF5 file for validation data.")
    parser.add_argument("--mode", type=str, choices=['sample_peaks', 'sample_dhs', 'all_dhs'], default='all_dhs', help="Mode of validation data generation.")
    parser.add_argument("--sample_ids", type=str, nargs='*', default=[], help="List of sample IDs to include in subset mode.")
    parser.add_argument("--dhs_ids", type=str, nargs='*', default=[], help="List of DHS IDs to include in subset mode.")
    args = parser.parse_args()

    anndata = read_zarr_backed(args.anndata_file)
    sample_ids = [sid for sid in args.sample_ids if check_none(sid)]
    dhs_ids = [did for did in args.dhs_ids if check_none(did)]

    if len(dhs_ids) == 0:
        assert len(sample_ids) > 0, "Either sample IDs or DHS IDs must be provided."
        if args.mode == 'sample_peaks':
            print('Generating validation data for sample peaks')
            # Logic to generate validation data for sample peaks
            data = generate_data_from_sample_peaks(anndata, sample_ids=sample_ids)
        else:
            use_sample_peaks = args.mode == 'sample_dhs'
            print('Generating validation data for sample DHSs')
            data = extract_dhs_data_from_backed_anndata(
                anndata,
                sample_ids=sample_ids,
                use_sample_peaks=use_sample_peaks
            )
    else:
        print('DHS IDs are provided. Ignoring "--mode" argument')
        data = extract_dhs_data_from_backed_anndata(
            anndata,
            sample_ids=sample_ids,
            dhs_ids=dhs_ids
        )
    data_to_h5(args.output_file, data)

    
