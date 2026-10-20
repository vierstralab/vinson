import argparse
import numpy as np

from genome_tools.data.anndata import read_zarr_backed
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp

from vinson.train_data.sampling import (
    delta_ball_exclusion_sampling,
    get_coo_mat,
    sample_cols_random,
    sample_jitter_offsets
)


def main(anndata, optimal_delta_radius, negatives_mult=2, seed=0, jitter=0):
    # calculate distance
    distance_matrix = squareform(
        pdist(
            anndata.obsm['motif_embeddings'],
            metric='cityblock'
        )
    )

    # get positives/negatives dense matrix
    binary = anndata.layers['binary'].toarray()

    # sample positives
    sampled_positives = delta_ball_exclusion_sampling(
        distance_matrix,
        delta_radius=optimal_delta_radius,
        mask=binary,
        seed=seed,
    )
    pos_coo_mat = get_coo_mat(sampled_positives, fill_value=1)

    n_positives = pos_coo_mat.sum(axis=0).A1

    # sample negatives
    sampled_negatives = delta_ball_exclusion_sampling(
        distance_matrix,
        delta_radius=optimal_delta_radius,
        mask=~binary,
        seed=seed,
    )
    neg_coo_mat = get_coo_mat(sampled_negatives, fill_value=-1)

    # subsample negatives to no more than negatives_mult of positives
    sampled_neg_coo_mat = sample_cols_random(
        neg_coo_mat.tocsc(),
        np.round(n_positives * negatives_mult).astype(int),
        seed=seed
    ).tocoo()

    class_mat = (pos_coo_mat + sampled_neg_coo_mat).tocsr()

    # jitter
    offsets_by_sample = sample_jitter_offsets(
        class_mat,
        anndata.obs_names,
        jitter,
        seed=seed
    )

    return class_mat, offsets_by_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample positive/negative training examples via delta-ball exclusion sampling"
    )
    parser.add_argument("anndata", type=str, help="Path to zarr-backed AnnData file")
    parser.add_argument("delta_radius", type=float, help="Optimal distance between selected samples")
    parser.add_argument("output", type=str, help="Path to save sampled examples (.npz file)")
    parser.add_argument(
        "jitter_output",
        type=str,
        help="Path to save per-sample jitter offsets (.npz file, keyed by sample_id)"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--negatives_mult", type=int, default=2, help="Multiplier of negatives relative to positives")
    parser.add_argument("--jitter", type=int, default=0, help="If >0, sample a random position offset in [-jitter, jitter] for each selected example")
    args = parser.parse_args()

    sampling_anndata = read_zarr_backed(args.anndata)

    class_mat, offsets_by_sample = main(
        sampling_anndata,
        args.delta_radius,
        negatives_mult=args.negatives_mult,
        seed=args.seed,
        jitter=args.jitter,
    )

    sp.save_npz(args.output, class_mat, compressed=True)
    np.savez(args.jitter_output, **offsets_by_sample)
