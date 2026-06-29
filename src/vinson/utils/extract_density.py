# Currently defunc

import numpy as np
import pandas as pd

import pyBigWig as pbw 
from genome_tools import GenomicInterval

from tqdm import tqdm

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
    bw_files: Iterable of str
        Paths to bigWig files containing per-sample density information.

    Attributes
    ----------
    bw_filehandles : list of pyBigWig.BigWig or None
        List of open bigWig file handles for each sample (initialized on first access).

    Methods
    -------
    __getitem__(x)
        Retrieve density values for all samples at a given genomic position or interval.
        Accepts either a (chrom, position) tuple or a GenomicInterval object.
        Returns pd.Series of density values
    __del__()
        Closes all open bigWig file handles on deletion.

    Notes
    -----
    - Density values are extracted using pyBigWig and returned as a pandas.Series.
    - Missing values are replaced with 0.0.
    - Designed for efficient batch extraction of per-sample genomic signal.
    """
    def __init__(self, bw_files):
        self.bw_files = pd.Series(bw_files)
        self.bw_filehandles = None

    def __del__(self):
        for fh in self.bw_filehandles:
            fh.close()

    def _init_filehandles(self):
        if self.bw_filehandles is None:
            self.bw_filehandles = [
                pbw.open(bw_file)
                for bw_file in self.bw_files
            ]
    def __getitem__(self, x):
        self._init_filehandles()
        
        if isinstance(x, tuple):
            chrom, mid = x
        elif isinstance(x, GenomicInterval):
            chrom = x.chrom
            mid = (x.start + x.end) // 2
        

        values = pd.Series(
            np.nan_to_num(
                [
                    fh.values(chrom, mid, mid + 1, numpy=True)[0]
                    for fh in tqdm(self.bw_filehandles)
                ],
                0.0,
            ),
            index=self.bw_files.index,
        )

        return values
