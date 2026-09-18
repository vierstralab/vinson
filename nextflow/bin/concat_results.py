import sys
import pandas as pd

files = sys.argv[2:]
outfile = sys.argv[1]



dfs = [pd.read_parquet(f) for f in files]



combined = pd.concat(dfs, ignore_index=True)



combined.to_parquet(outfile, index=False)
