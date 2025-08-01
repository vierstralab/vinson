import numpy as np
import pandas as pd

import pyBigWig as pbw 
from genome_tools import GenomicInterval

import logging
logger = logging.getLogger(__name__)


class SamplesDensityExtractor:
    """
    Extracts per-sample density values from a collection of bigWig files for genomic intervals.

    This class loads sample metadata from a tab-delimited file and opens bigWig files
    for each sample using a filepath pattern. It provides efficient access to density
    values at specified genomic positions or intervals for all samples.

    Parameters
    ----------
    samples_file : str
        Path to tab-delimited file containing sample metadata (used for column names).
    filepath_pattern : str
        Format string for bigWig file paths, with '{x}' replaced by sample name.

    Attributes
    ----------
    samples_df : pandas.DataFrame
        DataFrame containing sample metadata.
    bw_filehandles : list of pyBigWig.BigWig or None
        List of open bigWig file handles for each sample (initialized on first access).

    Methods
    -------
    __getitem__(x)
        Retrieve density values for all samples at a given genomic position or interval.
        Accepts either a (chrom, position) tuple or a GenomicInterval object.
        Returns a pandas.Series indexed by sample names.
    __del__()
        Closes all open bigWig file handles on deletion.

    Notes
    -----
    - Density values are extracted using pyBigWig and returned as a pandas.Series.
    - Missing values are replaced with 0.0.
    - Designed for efficient batch extraction of per-sample genomic signal.
    """
    def __init__(self, samples_file, filepath_pattern):
        self.filepath_pattern = filepath_pattern

        logger.info("Loading samples.")
        self.samples_df = pd.read_table(samples_file, index_col=0)

        self.bw_filehandles = None

    def __del__(self):
        for fh in self.bw_filehandles:
            fh.close()

    def __getitem__(self, x):
        if not self.bw_filehandles:
            self.bw_filehandles = [
                pbw.open(self.filepath_pattern.format(x=k))
                for k in self.samples_df.columns
            ]

        if isinstance(x, tuple):
            chrom, mid = x
        elif isinstance(x, GenomicInterval):
            chrom = x.chrom
            mid = (x.end - x.start) // 2 + x.start

        values = pd.Series(
            np.nan_to_num(
                [
                    fh.values(chrom, mid, mid + 1, numpy=True)[0]
                    for fh in self.bw_filehandles
                ],
                0.0,
            ),
            index=self.samples_df.columns,
        )

        return values
