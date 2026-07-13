"""Naive, unaligned distance metrics -- the deliberately WRONG tool.

These three functions compare two activation matrices R, R' index-for-index
(same instance row i, same neuron column j) with no rotation, no permutation,
no matching step of any kind. Notebook 3 uses them to show why Notebooks 1/2's
alignment-based metrics (Procrustes, Hungarian matching, CKA, RSA) were
necessary in the first place: naive index-matched comparison silently assumes
"neuron j in R is neuron j in R'", which is never guaranteed for two
independently-trained networks (there's no reason neuron j should end up
encoding the same thing in both).
"""

import numpy as np


def normalized_euclidean_distance(R, R_prime):
    """Scale each matrix to unit Frobenius norm independently, then return the
    Euclidean (Frobenius) distance between them. This removes pure scale
    differences between the two representations, but does NOTHING about neuron
    ordering -- column j of R and column j of R' are compared as if they were
    "the same neuron", which is an unjustified assumption for two independently
    trained networks.
    """
    A = R / np.linalg.norm(R, ord="fro")
    B = R_prime / np.linalg.norm(R_prime, ord="fro")
    return float(np.linalg.norm(A - B, ord="fro"))


def per_instance_cosine_similarity(R, R_prime):
    """For each test image i, cosine-similarity between R's and R''s full D-dim
    activation vector for that image, with no alignment. Returns (mean, std)
    across all N instances -- a distribution, not a single scalar, since
    per-instance behavior can vary a lot even when the average looks reasonable.
    """
    dot = np.sum(R * R_prime, axis=1)
    norms = np.linalg.norm(R, axis=1) * np.linalg.norm(R_prime, axis=1)
    norms = np.where(norms == 0, 1.0, norms)  # avoid 0/0 for all-dead-neuron rows
    cos_sim = dot / norms
    return float(cos_sim.mean()), float(cos_sim.std())


def per_neuron_cosine_similarity(R, R_prime):
    """For each neuron column j, cosine-similarity between R's and R''s length-N
    activation trace for that neuron, with no alignment. Returns (mean, std)
    across all D neurons.

    This is the metric most directly falsified by permutation: if clone neuron j
    actually corresponds to victim neuron k != j, comparing column j of R to
    column j of R' is meaningless by construction -- the permutation control
    experiment below proves exactly that.
    """
    dot = np.sum(R * R_prime, axis=0)
    norms = np.linalg.norm(R, axis=0) * np.linalg.norm(R_prime, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    cos_sim = dot / norms
    return float(cos_sim.mean()), float(cos_sim.std())
