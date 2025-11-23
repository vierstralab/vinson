import numpy as np
import pandas as pd
import argparse
import anndata as ad

from genome_tools.data.anndata import read_zarr_backed

from vinson.utils.data_formatting import extract_data_from_backed_anndata as extract_dhs_data_from_backed_anndata
from vinson.utils.data_formatting import data_to_h5, sanitize_data


def get_bg_for_peaks(peaks_df: pd.DataFrame, stats_path):
    stats = pd.read_table(stats_path).query(
        'fit_type == "segment"'
    ).rename(
        columns={'start': 'segment_start', 'end': 'segment_end'}
    )
    merged = peaks_df[['#chr', 'start', 'end', 'summit']].merge(
        stats, on="#chr", how="left"
    )
    merged['has_bg'] = merged.eval('summit >= segment_start & summit < segment_end')
    tmp = merged.groupby(['#chr', 'summit'])['has_bg'].max()
    peaks_without_bg = tmp[tmp == 0]
    if len(peaks_without_bg) > 0:
        print('No bg estimate at the summit')
        print(peaks_without_bg)
        non_merged = merged.set_index(['#chr', 'summit']).loc[peaks_without_bg.index]
        non_merged.query('start < segment_end & end > segment_start', inplace=True)
        assert len(non_merged) == len(peaks_without_bg), f"Could not find bg for all peaks without summit bg {len(non_merged)} vs {len(peaks_without_bg)}"
        non_merged['has_bg'] = True
        merged = pd.concat([merged.query('has_bg'), non_merged.reset_index()])

    merged = merged.query('has_bg').set_index(
        ['#chr', 'summit']
    ).loc[peaks_df.set_index(['#chr', 'summit']).index].reset_index()
    bg = merged.query('has_bg').eval('bg_r * bg_p / (1 - bg_p)').values
    assert len(bg) == len(peaks_df), f"Background length mismatch {len(bg)} vs {len(peaks_df)}"
    return bg


def generate_data_from_sample_peaks(anndata: ad.AnnData, sample_ids) -> dict:
    anndata_slice = anndata[sample_ids, :]
    data = []
    for sample_id, row in anndata_slice.obs.iterrows():
        sample_slice = anndata_slice[sample_id, :]
        fit_stats_file = row['hotspot3_fit_stats_file']
        peaks = pd.read_table(row['peaks_file_0.01fdr'])
        
        data_bundle = {
            'dhs_id': f'{sample_id}.' + peaks.index.astype(str).values,
            'sample_id': np.array([sample_id for _ in range(len(peaks))], dtype=np.str_),
            'read_depth': np.full(len(peaks), row['nuclear_reads'], dtype=np.float32),
            'chrom': peaks['#chr'].values,
            'summit': peaks['summit'].values,
            'background': get_bg_for_peaks(peaks, fit_stats_file),
            'class': np.ones(len(peaks), dtype=np.int8),
            'density': peaks['summit_density'].values,
        }
        if 'indiv_id' in anndata_slice.obsm:
            data_bundle['indiv_id'] = np.array([sample_slice.obsm['indiv_id'][0] for _ in range(len(peaks))], dtype=np.str_)
        else:
            print('Warning: indiv_id not found in anndata.obsm', flush=True)
        data.append(data_bundle)
    
    if len(data) > 1:
        data = {k: np.concatenate([d[k] for d in data]) for k in data[0].keys()}
    else:
        data = data[0]

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

    
