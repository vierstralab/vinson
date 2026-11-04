import sys
import pandas as pd


meta = pd.read_table(sys.argv[1])

outpath = sys.argv[2]
predictions_path = f'{outpath}/predictions/' + meta['prefix'] + '/' + meta['prefix'] + '.npy'
if 'dhs_dataset' not in meta.columns:
    meta['dhs_dataset'] = f'{outpath}/validation_data/' + meta['prefix'] + '.validation_data.h5'
meta['result_np'] = predictions_path
meta.to_csv(sys.argv[3], sep='\t', index=False)