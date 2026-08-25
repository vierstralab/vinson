from funkybob import RandomNameGenerator
import yaml
import gzip


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
    #makes sequence length too long if replaces char at start or end of seq
    return s[:i] + char + s[i + 1:]


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