import argparse
import numpy as np

from genome_tools.data.anndata import read_zarr_backed
import scipy.sparse as sp


def sample_jitter_offsets(class_mat, sample_names, jitter, seed=0):
    """
    For each sample (row of class_mat), sample a random position offset in [-jitter, jitter]
    for every DHS selected as a training example for that sample.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from sample_id to an (n_selected, 2) array of [dhs_index, offset].
    """
    rng = np.random.default_rng(seed=seed)
    offsets_by_sample = {}
    for i, sample_id in enumerate(sample_names):
        dhs_idx = class_mat.indices[class_mat.indptr[i]:class_mat.indptr[i + 1]]
        if dhs_idx.size == 0:
            continue
        offsets = rng.integers(-jitter, jitter + 1, size=dhs_idx.size)
        offsets_by_sample[str(sample_id)] = np.stack([dhs_idx, offsets], axis=1)
    return offsets_by_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample positive/negative training examples via delta-ball exclusion sampling"
    )
    parser.add_argument("sampled_matrix", type=str, help="Path to saved sampled examples (.npz sparse matrix)")
    parser.add_argument("sample_names", type=str, help="Path to corresponding sample names (rows)")
    parser.add_argument("output", type=str, help="Path to save per-sample jitter offsets (.npz file, keyed by sample_id)")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--jitter", type=int, default=200, help="If >0, sample a random position offset in [-jitter, jitter] for each selected example")
    args = parser.parse_args()

    sampling_anndata = read_zarr_backed(args.anndata)

    sampled_matrix = sp.load_npz(args.sampled_matrix)
    sample_names = np.loadtxt(args.sample_names, dtype=str)

    offsets_by_sample = sample_jitter_offsets(
        sampled_matrix,
        sample_names,
        seed=args.seed,
        jitter=args.jitter,
    )

    np.savez(args.jitter_output, **offsets_by_sample)
