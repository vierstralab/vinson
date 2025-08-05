import sys
import numpy as np
import h5py

from tqdm import tqdm

with h5py.File(sys.argv[1]) as f:
    chrom, pos, density, sample_id, bg = (
        f["chrom"][:],
        f["mid"][:],
        f["density"][:],
        f["sample_id"][:],
        f["bg_mu"][:],
    )

N = chrom.shape[0]
for i in tqdm(range(N)):
    out_line = "{}\t{}\t{}\t{}\t{:0.4f}\t{:0.4f}\n".format(
        chrom[i].astype(str), pos[i], pos[i] + 1, sample_id[i].astype(str), density[i], bg[i]
    )
    sys.stdout.write(out_line)
