import sys
import pandas as pd

outfile = sys.argv[1]
files = sys.argv[2:]

dfs = [pd.read_parquet(f) for f in files]

combined = pd.concat(dfs, ignore_index=True)

combined.to_parquet(outfile, index=False)