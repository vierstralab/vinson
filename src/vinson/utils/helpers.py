from funkybob import RandomNameGenerator
import yaml
import gzip
from typing import List
from genome_tools import GenomicInterval, VariantInterval


def generate_run_name() -> str:
    """Generate a timestamped random run name once on rank 0 and share it across ranks."""
    return next(iter(RandomNameGenerator()))


def read_yaml_config(path) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

def replace_at(s, i, char):
    """
    Replace the character at position `i` of `s` with `char` (e.g. injecting a SNP allele into a
    sequence at its relative position).

    Note: if `i` is at the very start/end of `s`, this appends `char` rather than overwriting,
    making the result one character longer than `s`.
    """
    return s[:i] + char + s[i + 1:]


def apply_edits(seq, interval: GenomicInterval, edits: List[VariantInterval], strict=True) -> str:
    """Copy seq, apply each VariantInterval edit (writes v.alt at v.start).
    Skips edits outside the window; strict-checks the reference allele."""
    seq = str(seq)
    assert len(seq) == len(interval), "Sequence length doesn't match length of the interval"
    for v in edits:
        rel = v.start - interval.start
        if rel <= 0 or rel >= len(seq):
            continue
        if strict and seq[rel:rel + len(v.ref)] != v.ref:
            raise AssertionError(f"ref mismatch {v.chrom}:{v.start} expected {v.ref}")
        seq = replace_at(seq, rel, v.alt)
    return seq


def detect_genotype_format(genotype_file):
    # helps with determining phased or unphased genotype file
    opener = gzip.open if genotype_file.endswith(".gz") else open

    with opener(genotype_file, "rt") as f:
        first_line = f.readline().strip()
        fields = first_line.split("\t")

        # header detection (your existing logic)
        try:
            int(fields[1])
            has_header = False
        except (ValueError, IndexError):
            has_header = True

        if has_header:
            first_line = f.readline().strip()
            fields = first_line.split("\t")

    n_cols = len(fields)

    if n_cols == 8:
        return True, has_header   # phased (has phase_block)
    elif n_cols == 7:
        return False, has_header  # unphased
    else:
        raise ValueError(f"Unexpected number of columns: {n_cols}")
