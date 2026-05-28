import sys
import pandas as pd


meta = pd.read_table(sys.argv[1])

outpath = sys.argv[2]
base_path = f'{outpath}/' + meta['prefix'] + '/' + meta['prefix']
if 'dhs_dataset' not in meta.columns:
    meta['dhs_dataset'] = f'{outpath}/validation_data/' + meta['prefix'] + '.validation_data.h5'
meta['result_np'] = base_path + '.npy'
meta.to_csv(sys.argv[3], sep='\t', index=False)