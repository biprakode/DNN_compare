from dnn_compare.activations import extract_everything
from dnn_compare.metrics import (
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
        report["representational"][layer_name] = {
            "svcca": svcca(R, R_prime),
            "orthogonal_procrustes": orthogonal_procrustes(R, R_prime),
            "soft_matching_distance": soft_matching_distance(R, R_prime),
            "cka_linear": cka_linear(R, R_prime),
            "rsa": rsa(R, R_prime),
            "knn_jaccard": knn_jaccard(R, R_prime),
            "magnitude": (magnitude(R), magnitude(R_prime)),
            "uniformity": (uniformity(R), uniformity(R_prime)),
            "intrinsic_dimension_pr": (intrinsic_dimension_pr(R), intrinsic_dimension_pr(R_prime)),
        }

    preds_a = everything_a["preds"]
    preds_b = everything_b["preds"]

    report["functional"] = {
        "performance_difference": performance_difference(preds_a, preds_b, labels),
        "cohens_kappa": cohens_kappa(preds_a, preds_b),
        "minmax_normalized_disagreement": minmax_normalized_disagreement(preds_a, preds_b, labels),
    }

    return report