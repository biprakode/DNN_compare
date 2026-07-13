"""The 12-metric comparison suite, per Klabunde et al., "Similarity of Neural
Network Models: A Survey of Functional and Representational Measures".

Representational metrics (1-9) operate on activation matrices R, R' of shape
(n_instances, d) extracted at the penultimate layer on the held-out test set.
Functional metrics (10-12) operate on hard predictions (+ ground truth) on the
same held-out set.

Where scipy/sklearn has a direct equivalent (Procrustes, Cohen's kappa, k-NN,
PCA, Hungarian assignment, accuracy) we call it directly rather than
reimplementing it. Only metrics with no library equivalent (CKA, RSA, soft
matching, min-max disagreement) are hand-rolled, and kept short.
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes as _orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Representational metrics (§3)
# --------------------------------------------------------------------------- #


def svcca(R, R_prime, variance_threshold=0.99):
    """§3.1 SVCCA. PCA-denoise both representations to `variance_threshold`
    explained variance (discards low-variance directions that are usually noise
    rather than signal in high-dim nets), then run CCA on the denoised
    representations and report the mean canonical correlation.
    """
    a = PCA(n_components=variance_threshold, svd_solver="full").fit_transform(R)
    b = PCA(n_components=variance_threshold, svd_solver="full").fit_transform(R_prime)
    n_components = min(a.shape[1], b.shape[1])
    a_c, b_c = CCA(n_components=n_components, max_iter=1000).fit_transform(a, b)
    corrs = [np.corrcoef(a_c[:, i], b_c[:, i])[0, 1] for i in range(n_components)]
    return float(np.mean(corrs))


def _center_unit_norm(R):
    """Center columns and rescale to unit Frobenius norm -- the standard
    normalization before an orthogonal (Procrustes) alignment, so the resulting
    distance is a bounded, scale-invariant quantity rather than raw activation units.
    """
    C = R - R.mean(axis=0)
    return C / np.linalg.norm(C, ord="fro")


def orthogonal_procrustes(R, R_prime):
    """§3.2 Procrustes. Closed-form optimal orthogonal (rotation/reflection)
    alignment between the two representations -- invariant to any orthogonal
    transform of the representation space, which is the natural nuisance symmetry
    when comparing two independently-trained networks. Uses
    scipy.linalg.orthogonal_procrustes directly (it exists; no need to hand-roll).
    """
    A = _center_unit_norm(R)
    B = _center_unit_norm(R_prime)
    Q, _ = _orthogonal_procrustes(A, B)
    return float(np.linalg.norm(A @ Q - B))


def soft_matching_distance(R, R_prime):
    """§3.2 Soft matching distance. Restricts alignment to a 1-1 neuron PERMUTATION
    rather than a full rotation, solved optimally via the Hungarian algorithm on a
    neuron-by-neuron distance matrix. Note: with D == D' in this setup, this is
    close to a permutation-restricted Procrustes and mostly serves as a sanity
    check against `orthogonal_procrustes` here -- its real advantage shows up when
    comparing representations of different widths, which isn't the case here.
    """
    A = _center_unit_norm(R)
    B = _center_unit_norm(R_prime)
    cost = cdist(A.T, B.T, metric="euclidean")  # cost[i, j]: neuron i (R) vs neuron j (R')
    row_idx, col_idx = linear_sum_assignment(cost)
    return float(cost[row_idx, col_idx].mean())


def cka_linear(R, R_prime):
    """§3.3 Linear CKA -- the primary representational-similarity metric in this
    pipeline. Invariant to orthogonal transform AND isotropic scaling (a strictly
    weaker, more forgiving invariance than Procrustes' rotation-only invariance).

    Implemented via the compact feature-covariance form (centered features, then
    ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)), which is mathematically identical
    to the textbook HSIC/Gram-matrix definition for a linear kernel but O(n*d^2)
    instead of O(n^2) -- avoids building a 10000x10000 kernel matrix for our
    held-out set.
    """
    Xc = R - R.mean(axis=0)
    Yc = R_prime - R_prime.mean(axis=0)
    cross = Yc.T @ Xc
    numerator = np.linalg.norm(cross, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(numerator / denom)


def rsa(R, R_prime):
    """§3.3 RSA. Build each model's Representational Similarity Matrix (pairwise
    instance distances), then Spearman-correlate the two RSMs' lower triangles.
    Invariant to ANY transform that preserves relative pairwise distances -- doesn't
    even require the two representations to share a coordinate system, unlike
    Procrustes/CKA. Cast to float32 to keep the two 10000x10000 distance matrices
    memory-manageable.
    """
    rsm_a = squareform(pdist(R.astype(np.float32), metric="euclidean"))
    rsm_b = squareform(pdist(R_prime.astype(np.float32), metric="euclidean"))
    tri = np.triu_indices_from(rsm_a, k=1)
    corr, _ = spearmanr(rsm_a[tri], rsm_b[tri])
    return float(corr)


def knn_jaccard(R, R_prime, k=10):
    """§3.4 kNN-Jaccard. For each instance, take its k-nearest-neighbor set in each
    representation and report the mean Jaccard overlap. A much weaker requirement
    than CKA/Procrustes/RSA: only asks "are the same things nearby", not "is the
    global geometry the same".
    """
    idx_a = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R, return_distance=False)
    idx_b = NearestNeighbors(n_neighbors=k + 1).fit(R_prime).kneighbors(R_prime, return_distance=False)
    jaccards = np.empty(R.shape[0])
    for i in range(R.shape[0]):
        set_a = set(idx_a[i, 1:])  # column 0 is always "self"; drop it
        set_b = set(idx_b[i, 1:])
        jaccards[i] = len(set_a & set_b) / len(set_a | set_b)
    return float(jaccards.mean())


def magnitude(R):
    """§3.6 Magnitude. Simplest single-representation statistic: norm of the mean
    instance vector. Called once per model; the pipeline reports the pair plus
    their absolute difference. Useful as a cheap sanity check for systematic scale
    shifts that CKA (scale-invariant by design) would hide entirely.
    """
    return float(np.linalg.norm(R.mean(axis=0)))


def uniformity(R, t=2):
    """§3.6 Uniformity (Eq. 46). log of the mean pairwise Gaussian-kernel
    similarity across all instance pairs -- a single-representation statistic
    (called once per model). Lower (more negative) means the representation
    spreads its instances more uniformly over the space rather than clumping.

    Rows are L2-normalized onto the unit hypersphere first -- this is how the
    metric is defined in its original context (Wang & Isola 2020, contrastive
    embeddings live on a unit sphere). Without it, `-t * ||x-y||^2` scales with
    the representation's raw activation magnitude: on our unnormalized-pixel-scale
    activations that exponent underflows to ~0 and log(~0) blows up to a huge
    negative number that's really just measuring input scale, not uniformity.

    A handful of test images can produce an all-zero penultimate-layer row (every
    ReLU unit off for that input, which is more likely here given how many dead
    neurons we found earlier on unnormalized-pixel inputs) -- dividing those rows
    by a zero norm would produce NaN and poison the whole mean, so those rows are
    left at the origin (0/1 = 0) instead of blowing up to NaN.
    """
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    R_normed = R / norms
    sq_dists = pdist(R_normed.astype(np.float32), metric="sqeuclidean")
    return float(np.log(np.mean(np.exp(-t * sq_dists))))


def intrinsic_dimension_pr(R):
    """§3.6 Intrinsic dimension via the Participation Ratio estimator:
    (sum(eigenvalues))^2 / sum(eigenvalues^2), computed on the feature covariance.
    A simple, standard proxy for effective dimensionality -- deliberately not a
    heavier estimator (e.g. TwoNN) per the spec, since participation ratio is
    stable enough for an 84-dim representation.
    """
    Rc = R - R.mean(axis=0)
    cov = np.cov(Rc, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), a_min=0, a_max=None)
    return float((eigvals.sum() ** 2) / (eigvals ** 2).sum())


# --------------------------------------------------------------------------- #
# Functional metrics (§4)
# --------------------------------------------------------------------------- #


def performance_difference(preds_victim, preds_clone, y_true):
    """§4.1 Performance difference. The simplest possible functional comparison:
    absolute gap in task accuracy between victim and clone on the same held-out set.
    Uses sklearn.metrics.accuracy_score directly.
    """
    acc_v = accuracy_score(y_true, preds_victim)
    acc_c = accuracy_score(y_true, preds_clone)
    return float(abs(acc_v - acc_c))


def cohens_kappa(preds_victim, preds_clone):
    """§4.2 Cohen's kappa. Chance-corrected agreement between victim's and clone's
    PREDICTED labels (not vs. ground truth) -- captures whether they make the same
    mistakes, which raw accuracy alone can't distinguish. Uses
    sklearn.metrics.cohen_kappa_score directly.
    """
    return float(cohen_kappa_score(preds_victim, preds_clone))


def minmax_normalized_disagreement(preds_victim, preds_clone, y_true):
    """§4.2 Min-max normalized disagreement (Eq. 53). Rescales raw disagreement
    (fraction of instances where victim and clone predict differently) between the
    theoretical MINIMUM possible disagreement (|err_victim - err_clone|, forced by
    each model's own error rate even if they were "as agreeing as possible") and the
    theoretical MAXIMUM (min(err_victim + err_clone, 1)). This stops two clones with
    different accuracies from being penalized just for having more room to disagree.
    """
    raw_disagreement = float(np.mean(preds_victim != preds_clone))
    err_v = 1.0 - accuracy_score(y_true, preds_victim)
    err_c = 1.0 - accuracy_score(y_true, preds_clone)
    min_possible = abs(err_v - err_c)
    max_possible = min(err_v + err_c, 1.0)
    if max_possible - min_possible < 1e-12:
        return 0.0
    return float((raw_disagreement - min_possible) / (max_possible - min_possible))
