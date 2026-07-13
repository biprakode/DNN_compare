import numpy as np
import torch

from dnn_compare.activations import extract_everything
from dnn_compare.metrics import (
    agreement,
    cka_linear,
    cohens_kappa,
    intrinsic_dimension_pr,
    knn_jaccard,
    magnitude,
    minmax_normalized_disagreement,
    orthogonal_procrustes,
    performance_difference,
    rsa,
    soft_matching_distance,
    svcca,
    uniformity,
)
from dnn_compare.models import NN
from dnn_compare.naive_metrics import normalized_euclidean_distance, per_instance_cosine_similarity, per_neuron_cosine_similarity


def compute_full_report(model_a : NN , model_b : NN , images, labels) -> dict:
    everything_a = extract_everything(model_a , images)
    everything_b = extract_everything(model_b , images)
    return _build_report(everything_a, everything_b, labels)


def compute_full_report_with_progress(model_a : NN , model_b : NN , images, labels, chunk_size: int = None, on_progress=None) -> dict:
    """Same result as compute_full_report, but runs both models' forward
    passes in lockstep chunks over `images` instead of one vectorized batch,
    calling on_progress(images_tested, total_images) after each chunk. Lets a
    caller (e.g. an API endpoint) report real progress on the actual
    forward-pass work, not a simulated timer.
    """
    n = images.shape[0]
    if chunk_size is None:
        chunk_size = max(1, n // 10)

    chunks_a: dict[str, list] = {}
    chunks_b: dict[str, list] = {}

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = images[start:end]

            out_a, preds_a = model_a(batch, return_all_layers=True)
            out_a["preds"] = preds_a
            for key, tensor in out_a.items():
                chunks_a.setdefault(key, []).append(tensor.cpu().numpy())

            out_b, preds_b = model_b(batch, return_all_layers=True)
            out_b["preds"] = preds_b
            for key, tensor in out_b.items():
                chunks_b.setdefault(key, []).append(tensor.cpu().numpy())

            if on_progress is not None:
                on_progress(end, n)

    everything_a = {key: np.concatenate(v, axis=0) for key, v in chunks_a.items()}
    everything_b = {key: np.concatenate(v, axis=0) for key, v in chunks_b.items()}
    return _build_report(everything_a, everything_b, labels)


def _build_report(everything_a: dict, everything_b: dict, labels) -> dict:
    layer_names = [k for k in everything_a if k != "preds"]

    report = {"naive": {}, "representational": {}, "functional": {}}

    for layer_name in layer_names:
        R , R_prime = everything_a[layer_name] , everything_b[layer_name]
        report["naive"][layer_name] = {
            "normalized_euclidean_distance": normalized_euclidean_distance(R, R_prime),
            "per_instance_cosine_similarity": per_instance_cosine_similarity(R, R_prime),
            "per_neuron_cosine_similarity": per_neuron_cosine_similarity(R, R_prime),
        }

    for layer_name in layer_names:
        R , R_prime = everything_a[layer_name] , everything_b[layer_name]
        rsa_corr, victim_rsm, clone_rsm = rsa(R, R_prime)
        report["representational"][layer_name] = {
            "svcca": svcca(R, R_prime),
            "orthogonal_procrustes": orthogonal_procrustes(R, R_prime),
            "soft_matching_distance": soft_matching_distance(R, R_prime),
            "cka_linear": cka_linear(R, R_prime),
            "rsa": rsa_corr,
            "victim_rsm": victim_rsm.tolist(),
            "clone_rsm": clone_rsm.tolist(),
            "knn_jaccard": knn_jaccard(R, R_prime),
            "magnitude": (magnitude(R), magnitude(R_prime)),
            "uniformity": (uniformity(R), uniformity(R_prime)),
            "intrinsic_dimension_pr": (intrinsic_dimension_pr(R), intrinsic_dimension_pr(R_prime)),
        }

    preds_a = everything_a["preds"]
    preds_b = everything_b["preds"]

    perf_diff, accuracy_victim, accuracy_clone = performance_difference(preds_a, preds_b, labels)

    report["functional"] = {
        "performance_difference": perf_diff,
        "accuracy_victim": accuracy_victim,
        "accuracy_clone": accuracy_clone,
        "agreement": agreement(preds_a, preds_b),
        "cohens_kappa": cohens_kappa(preds_a, preds_b),
        "minmax_normalized_disagreement": minmax_normalized_disagreement(preds_a, preds_b, labels),
    }

    return report