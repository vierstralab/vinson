"""
Convert per-sample footprint tabix files into bigWigs, {outdir}/{sample_id}.obs.bw and
{outdir}/{sample_id}.exp.bw, and optionally merge them into one figwig file.

Usage:
    python fp_tabix_to_bigwig.py <fp_meta_file> <genome.fa.fai> <outdir> [--figwig out.fw] [--n_jobs 8]

fp_meta_file: tab-separated, with columns sample_id and tabix_file.
`figwig bw2fw` names each track after its file, so tracks are <sample_id>.obs and
<sample_id>.exp, as SequenceEmbedProfileDataset expects.
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pysam

from figwig import BigWigWriter
from genome_tools import GenomicInterval
from genome_tools.data.extractors import TabixExtractor

FP_COLUMNS = ['chrom', 'start', 'end', 'exp', 'obs', 'neglog_pval', 'neglog_winpval', 'fdr']


def tabix_to_bigwigs(fp_tabix, chrom_sizes, out_prefix):
    present = set(pysam.TabixFile(fp_tabix).contigs)

    with BigWigWriter(f"{out_prefix}.obs.bw", chrom_sizes) as obs_writer, \
            BigWigWriter(f"{out_prefix}.exp.bw", chrom_sizes) as exp_writer, \
            TabixExtractor(fp_tabix, columns=FP_COLUMNS) as ext:
        # writers take items in chrom_sizes order
        for chrom, size in chrom_sizes.items():
            if chrom not in present:
                continue
            df = ext[GenomicInterval(chrom, 0, size)]
            if len(df) == 0:
                continue
            starts = df['start'].values.astype(np.int64)
            ends = df['end'].values.astype(np.int64)
            obs_writer.write(chrom, starts, df['obs'].values.astype(np.float32), ends=ends)
            exp_writer.write(chrom, starts, df['exp'].values.astype(np.float32), ends=ends)
    return out_prefix


def main(fp_meta_file, fai, outdir, figwig_file=None, n_jobs=1):
    fp_meta = pd.read_table(fp_meta_file)
    chrom_sizes = pd.read_table(fai, header=None, usecols=[0, 1], index_col=0).iloc[:, 0].to_dict()
    os.makedirs(outdir, exist_ok=True)

    prefixes = [os.path.join(outdir, str(s)) for s in fp_meta['sample_id']]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for prefix in pool.map(tabix_to_bigwigs, fp_meta['tabix_file'], [chrom_sizes] * len(prefixes), prefixes):
            print(f"wrote {prefix}.obs.bw, {prefix}.exp.bw", flush=True)

    if figwig_file is not None:
        bigwigs = [f"{p}.{kind}.bw" for p in prefixes for kind in ('obs', 'exp')]
        subprocess.run(['figwig', 'bw2fw', '-o', figwig_file, '-p', str(n_jobs), *bigwigs], check=True)
        print(f"wrote {figwig_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fp_meta_file', help='Tab-separated file with columns sample_id and tabix_file')
    parser.add_argument('fai', help='Genome .fai (chromosome sizes, in the order to write)')
    parser.add_argument('outdir', help='Directory for the per-sample bigWigs')
    parser.add_argument('--figwig', default=None, help='If given, merge all bigWigs into this figwig file')
    parser.add_argument('--n_jobs', type=int, default=1, help='Samples converted in parallel (and bw2fw threads)')
    args = parser.parse_args()

    main(args.fp_meta_file, args.fai, args.outdir, figwig_file=args.figwig, n_jobs=args.n_jobs)
