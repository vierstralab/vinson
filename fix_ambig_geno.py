#!/usr/bin/env python

import gzip
from tqdm import tqdm

########################
# Paths
########################
GENO_IN = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/phasing/output/all_phased.bed.gz"
GENO_OUT = "/net/seq/data2/projects/mbrannon/all_phased.no_ambiguous.bed.gz"
AMBIG_OUT = "/net/seq/data2/projects/mbrannon/ambiguous_sites.bed"

########################
# Counters
########################
ambiguous = 0
kept = 0

########################
# Stream + filter
########################
print("Filtering ambiguous genotype sites…")

with gzip.open(GENO_IN, "rt") as fin, \
     gzip.open(GENO_OUT, "wt") as fout, \
     open(AMBIG_OUT, "w") as famb:

    header = fin.readline()
    fout.write(header)
    famb.write(header)

    for line in tqdm(fin, desc="Genotypes"):
        fields = line.rstrip().split()
        ref = fields[3]
        alt = fields[4]

        if len(ref) != 1 or len(alt) != 1 or ref == alt:
            ambiguous += 1
            famb.write(line)
        else:
            kept += 1
            fout.write(line)

########################
# Report
########################
print("========== FILTER REPORT ==========")
print(f"Kept sites:      {kept:,}")
print(f"Ambiguous sites: {ambiguous:,}")
print(f"Output written to:")
print(f"  Cleaned BED:   {GENO_OUT}")
print(f"  Ambiguous BED: {AMBIG_OUT}")
