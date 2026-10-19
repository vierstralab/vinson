import argparse
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import numba
from genome_tools.data.anndata import read_zarr_backed
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp


def delta_ball_exclusion_sampling(
        distance_matrix,
        delta_radius,
        size,
        mask=None,
        max_samples=None,
        seed=0,
    ):
    """
    Thin wrapper of numba delta-ball exclusion sampling of samples based on a distance matrix.

    Parameters
    ----------
    distance_matrix : np.array
        Square (n_samples x n_samples) distance matrix.
    delta_radius : float
        Minimum distance between selected samples.
    size : int
        Number of initial sample indices to use for seeding the sampling.
    mask : array-like of bool, shape (n_samples, size), optional
        Mask indicating which samples can be selected for each initial index.
        If None, all samples are allowed.
    max_samples : int, optional
        Maximum number of samples to select. If None, no limit is applied (max_samples = n_samples).
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (size, n_samples) with boolean values indicating selected samples.
    """
    if max_samples is None:
        max_samples = distance_matrix.shape[0]
    max_samples = np.asarray(max_samples)
    if len(max_samples.shape) == 0:
        max_samples = np.full(size, max_samples, dtype=int)
    
    assert max_samples.shape == (size,)
    assert (max_samples <= distance_matrix.shape[0]).all()
    

    if mask is None:
        mask = np.ones((distance_matrix.shape[0], size), dtype=bool)

    assert mask.shape == (distance_matrix.shape[0], size), "Mask shape must be (n_samples, size)"

    samples = fast_delta_ball_exclusion_sampling(
        distance_matrix, mask, delta_radius, max_samples, seed
    )
    return samples


@numba.njit
def weighted_choice_numba(population, weights):
    r = np.random.random()
    acc = 0.0
    for i in range(weights.shape[0]):
        acc += weights[i]
        if r < acc:
            return population[i]
    return population[weights.shape[0] - 1]


@numba.njit(parallel=True)
def fast_delta_ball_exclusion_sampling(D2, mask, delta_radius, max_samples, seed):
    """
    Delta-ball exclusion sampling.
    Samples points such that no two points are within delta_radius distance.

    Parameters
    ----------
    D2 : ndarray
        Square (N x N) distance matrix.

    mask : ndarray
        Boolean mask of shape (N,) indicating which samples can be selected.

    delta_radius : float
        Minimum distance between selected samples.

    max_samples : int
        Maximum number of samples to select.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    selected_sample_indices : ndarray
        Boolean array indicating which samples were selected.

    """
    N = D2.shape[0]
    S = mask.shape[1]
    selected_sample_indices = np.zeros((S, N), dtype=np.bool_)

    for s in numba.prange(S):
        pickable_weights = mask[:, s].copy()
        max_total = min(max_samples[s], pickable_weights.sum()) # Total number of samples to pick

        for i in range(max_total):
            total_weight = pickable_weights.sum()

            if total_weight == 0:
                break

            #TODO: upgrade numa to 0.60 and fi rng in parallel
            np.random.seed(s*12345 + i*123 + seed)
            pick_index = weighted_choice_numba(
                np.arange(N),
                pickable_weights / total_weight,
            )
            selected_sample_indices[s, pick_index] = True
            pickable_weights[D2[pick_index, :] < delta_radius] = 0  # Exclude samples within delta_radius of already picked samples

    return selected_sample_indices


def sample_cols_random(
        X: csc_matrix,
        m_per_col: np.ndarray,
        sampled_reference: csc_matrix = None,
        distance_matrix: np.ndarray = None,
        seed=0
    ) -> csc_matrix:
    """
    Randomly keep at most m_per_col[j] nonzero entries per column of a CSC sparse matrix.
    Sampling is uniform without replacement within each column.

    If sampled_reference and distance_matrix are provided,
    the sampling probabilities are inversely proportional to the minimum distance to reference points, i.e. sample furthest points from sampled_reference e.g. negatives.

    Parameters
    ----------
    X : csc_matrix
        Input sparse matrix in CSC format. Shape (n_samples, n_dhss).
    m_per_col : np.ndarray
        1D array of length n_cols specifying the maximum number of nonzero entries to keep per column.
    sampled_reference : csc_matrix, optional
        Reference sparse matrix in CSC format. Shape (n_samples, n_dhss).
        If provided, sampling probabilities are based on minimal distance to one of these reference points.
    distance_matrix : np.ndarray, optional
        2D array of shape (n_samples, n_samples) specifying pairwise distances between samples.
    """
    rng = np.random.default_rng(seed=seed)
    X = X.tocsc(copy=True)
    nrows, ncols = X.shape
    assert len(m_per_col) == ncols, "m_per_col length must equal number of columns"

    use_weights = False
    if sampled_reference is not None:
        assert sampled_reference.shape == X.shape, "sampled_reference shape must match X shape"
        assert distance_matrix is not None, "distance_matrix must be provided when sampled_reference is provided"
        assert distance_matrix.shape[0] == nrows and distance_matrix.shape[1] == nrows, "distance_matrix must be square with size equal to number of rows in X"
        use_weights = True

    data, indices, indptr = [], [], [0]
    for j in range(ncols):
        start, end = X.indptr[j], X.indptr[j+1]
        k = end - start
        m = min(m_per_col[j], k)
        if m > 0:
            if use_weights:
                ref_start, ref_end = sampled_reference.indptr[j], sampled_reference.indptr[j+1]
                ref_rows = sampled_reference.indices[ref_start:ref_end]

                # compute weights
                eps = 1e-12
                # distance from each active row to the *nearest* reference row
                d = distance_matrix[np.ix_(X.indices[start:end], ref_rows)]  # shape (k, s)
                d_min = d.min(axis=1)
                w = 1.0 / (d_min + eps)

                p = w / w.sum()
            else:
                p = None
            sel = rng.choice(k, m, replace=False, p=p)
            data.append(X.data[start + sel])
            indices.append(X.indices[start + sel])
        indptr.append(indptr[-1] + m)

    return csc_matrix(
        (
            np.concatenate(data),
            np.concatenate(indices),
            np.array(indptr)
        ),
        shape=X.shape
    )


def get_coo_mat(sampled, fill_value=1, dtype=np.int8):
    return coo_matrix(sampled.T).astype(dtype) * fill_value


def sample_jitter_offsets(class_mat, obs_names, jitter, seed=0):
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
    for i, sample_id in enumerate(obs_names):
        dhs_idx = class_mat.indices[class_mat.indptr[i]:class_mat.indptr[i + 1]]
        if dhs_idx.size == 0:
            continue
        offsets = rng.integers(-jitter, jitter + 1, size=dhs_idx.size)
        offsets_by_sample[str(sample_id)] = np.stack([dhs_idx, offsets], axis=1)
    return offsets_by_sample


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
    offsets_by_sample = sample_jitter_offsets(class_mat, anndata.obs_names, jitter, seed=seed)

    return class_mat, offsets_by_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample positive/negative training examples via delta-ball exclusion sampling")
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
