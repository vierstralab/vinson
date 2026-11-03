"""
Write per-sample footprint tabix files into one figwig file, with tracks <sample_id>.obs and
<sample_id>.exp, as SequenceEmbedProfileDataset reads them.

Usage:
    python fp_tabix_to_figwig.py <fp_meta_file> <chrom_sizes> <out.fw> [--n_jobs 8] [--tile 16384]

fp_meta_file: tab-separated, with columns sample_id and tabix_file.

Each sample's obs/exp pair is its own track group, so training reads (one sample's two tracks per
example) decode only that pair. Samples are written one at a time, chromosome by chromosome in chrom sizes
order; tabix files are parsed in parallel worker processes.
"""
import argparse
import itertools
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pysam

from figwig import FigWigWriter
from genome_tools import GenomicInterval
from genome_tools.data.extractors import TabixExtractor

FP_COLUMNS = ['chrom', 'start', 'end', 'exp', 'obs', 'neglog_pval', 'neglog_winpval', 'fdr']


def read_chrom(fp_tabix, chrom, size):
    if chrom not in pysam.TabixFile(fp_tabix).contigs:
        return None
    with TabixExtractor(fp_tabix, columns=FP_COLUMNS) as ext:
        df = ext[GenomicInterval(chrom, 0, size)]
    if len(df) == 0:
        return None
    values = np.stack([df['obs'].values, df['exp'].values], axis=1).astype(np.float32)
    return df['start'].values.astype(np.int64), df['end'].values.astype(np.int64), values


def main(fp_meta_file, chrom_sizes_file, out_fw, n_jobs=1, tile=None):
    fp_meta = pd.read_table(fp_meta_file)
    chrom_sizes = pd.read_table(chrom_sizes_file, header=None, names=['chrom', 'length'], sep=r'\s+').set_index('chrom')['length'].to_dict()

    sample_ids = [str(s) for s in fp_meta['sample_id']]
    tracks = [
        {"name": f"{s}.{kind}", "sample_id": s, "kind": kind}
        for s in sample_ids for kind in ('obs', 'exp')
    ]
    groups = [[f"{s}.obs", f"{s}.exp"] for s in sample_ids]

    # sample-major task order: within a group, chromosomes must come in order, and a group
    # cannot return to a chromosome it has left
    tasks = [
        (sample_id, fp_tabix, chrom, size)
        for sample_id, fp_tabix in zip(sample_ids, fp_meta['tabix_file'])
        for chrom, size in chrom_sizes.items()
    ]

    last_chrom = list(chrom_sizes)[-1]
    with FigWigWriter(out_fw, chrom_sizes, tracks, groups=groups, tile=tile) as writer, \
            ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # bounded lookahead: parsed chromosomes wait in memory until written in task order
        pending = deque()
        tasks_iter = iter(tasks)
        for task in itertools.islice(tasks_iter, 2 * n_jobs):
            pending.append((task, pool.submit(read_chrom, *task[1:])))
        while pending:
            (sample_id, _, chrom, _), future = pending.popleft()
            result = future.result()
            next_task = next(tasks_iter, None)
            if next_task is not None:
                pending.append((next_task, pool.submit(read_chrom, *next_task[1:])))
            if result is not None:
                starts, ends, values = result
                writer.write(chrom, starts, values, ends=ends, tracks=[f"{sample_id}.obs", f"{sample_id}.exp"])
            if chrom == last_chrom:
                print(f"wrote {sample_id}", flush=True)
    print(f"wrote {out_fw}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fp_meta_file', help='Tab-separated file with columns sample_id and tabix_file')
    parser.add_argument('chrom_sizes_file', help='Two-column chrom sizes file (chrom, length), in the order to write')
    parser.add_argument('out_fw', help='Output figwig file')
    parser.add_argument('--n_jobs', type=int, default=1, help='Processes parsing tabix files in parallel')
    parser.add_argument('--tile', type=int, default=None, help='figwig tile size (default 16384); smaller can read faster on one thread')
    args = parser.parse_args()

    main(args.fp_meta_file, args.chrom_sizes_file, args.out_fw, n_jobs=args.n_jobs, tile=args.tile)
