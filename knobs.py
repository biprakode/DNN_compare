import numpy as np


def kaiming_uniform_like_torch(fan_in: int, size: tuple, rng: np.random.Generator) -> np.ndarray:
    bound = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-bound, bound, size=size)


def apply_signature_knob(
        weight_matrix: np.ndarray,
        unrecovered_count: int,
        seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    out_features, in_features = weight_matrix.shape
    if not (0 <= unrecovered_count <= out_features):
        raise ValueError(
            f"unrecovered_count={unrecovered_count} out of range "
            f"for {out_features} neurons in this layer"
        )

    rng = np.random.default_rng(seed)
    unrecovered_idx = rng.choice(out_features, size=unrecovered_count, replace=False)

    recovered_mask = np.ones(out_features, dtype=bool)
    recovered_mask[unrecovered_idx] = False   # False = fully reinitialized

    mutated_matrix = weight_matrix.copy()
    mutated_matrix[unrecovered_idx] = kaiming_uniform_like_torch(
        fan_in=in_features,
        size=(unrecovered_count, in_features),
        rng=rng,
    )
    return mutated_matrix, recovered_mask


def apply_cosine_knob(
        weight_matrix: np.ndarray,
        recovered_mask: np.ndarray,
        target_cosine: float,
        seed: int,
) -> np.ndarray:
    """Rotates each recovered-mask row toward target_cosine similarity with its
    original direction, relative to the ORIGINAL (pre-sign-flip) row -- the
    sign knob is applied after this, per the resolved rotate-then-flip order.
    """
    rng = np.random.default_rng(seed)
    mutated_matrix = weight_matrix.copy()
    recovered_idx = np.flatnonzero(recovered_mask)

    for i in recovered_idx:
        row = mutated_matrix[i]
        row_norm = np.linalg.norm(row)
        if row_norm == 0:
            continue
        unit_row = row / row_norm

        random_vec = rng.standard_normal(row.shape[0])
        orthogonal = random_vec - np.dot(random_vec, unit_row) * unit_row
        orthogonal_norm = np.linalg.norm(orthogonal)
        if orthogonal_norm == 0:
            continue
        unit_orthogonal = orthogonal / orthogonal_norm

        new_unit = target_cosine * unit_row + np.sqrt(max(0.0, 1 - target_cosine ** 2)) * unit_orthogonal
        mutated_matrix[i] = new_unit * row_norm

    return mutated_matrix


def apply_sign_knob(
        weight_matrix: np.ndarray,
        sign_flip_count: int,
        seed: int,
) -> np.ndarray:
    """Independent of the signature knob's recovered_mask by design: selects
    sign_flip_count rows uniformly at random from the FULL set of out_features
    (no filtering), using its own seed, and negates each selected row.
    """
    out_features, _ = weight_matrix.shape
    if not (0 <= sign_flip_count <= out_features):
        raise ValueError(
            f"sign_flip_count={sign_flip_count} out of range "
            f"for {out_features} neurons in this layer"
        )

    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(out_features, size=sign_flip_count, replace=False)

    mutated_matrix = weight_matrix.copy()
    mutated_matrix[flip_idx] *= -1
    return mutated_matrix


def apply_layer_knobs(
        weight_matrix: np.ndarray,
        unrecovered_count: int,
        target_cosine: float,
        sign_flip_count: int,
        signature_seed: int,
        sign_seed: int,
) -> np.ndarray:
    """Orchestrates all three knobs for one layer's weight matrix.

    Ordering (resolved): the cosine knob rotates recovered rows toward
    target_cosine relative to their ORIGINAL direction first; the sign knob
    then flips its independently-selected rows on the result. A row can be
    touched by the signature knob, the cosine knob, the sign knob, any
    combination, or none -- the sign knob's selection is fully independent of
    recovered_mask and uses its own seed.
    """
    mutated_matrix, recovered_mask = apply_signature_knob(
        weight_matrix, unrecovered_count, signature_seed
    )

    if target_cosine < 1.0:
        mutated_matrix = apply_cosine_knob(
            mutated_matrix, recovered_mask, target_cosine, signature_seed
        )

    if sign_flip_count > 0:
        mutated_matrix = apply_sign_knob(
            mutated_matrix, sign_flip_count, sign_seed
        )

    return mutated_matrix
