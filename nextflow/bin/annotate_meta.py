import sys
import pandas as pd



meta = pd.read_table(sys.argv[1])
outpath = sys.argv[2]
meta['result_np'] = outpath + '/' + meta['prefix'] + '.npy'
meta.to_csv(sys.argv[3], sep='\t', index=False)