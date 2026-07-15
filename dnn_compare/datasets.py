import os

import numpy as np
import torch
from sklearn.datasets import make_blobs as _make_blobs

CIFAR_FLATTENED_WIDTH = 32 * 32 * 3
# Real CIFAR-10 already exists on this machine (fetched previously for a
# Keras run) -- point straight at it instead of re-downloading the ~170MB
# archive over this sandbox's very slow connection.
_CIFAR_LOCAL_DATASET_DIR = "/home/biprarshi/.keras/datasets/cifar-10-batches-py-target"
_MAKEBLOBS_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tiny_stuff", "makeblobs_testdata")

SUPPORTED_DATASETS = ("make_blobs", "cifar")

# The reference victims in tiny_stuff/ were each trained on their OWN
# make_blobs draw (different seed/cluster_std per architecture) and then
# StandardScaler-fit + clip(-3,3)/3.0 -- NOT raw make_blobs output. Generating
# fresh synthetic data on the fly (as this module used to do unconditionally)
# gives these victims near-chance accuracy since the input distribution
# doesn't match what they were trained on at all. For these three known
# input widths we instead serve the exact preprocessed test set they were
# evaluated on; other widths (a custom-trained checkpoint) fall back to
# synthetic generation, since there's no ground-truth test set to serve there.
_MAKEBLOBS_TESTDATA_BY_WIDTH = {
    8: ("x_test_tiniest_makeblobs.npy", "y_test_tiniest_makeblobs.npy"),
    32: ("x_test_tinier_makeblobs.npy", "y_test_tinier_makeblobs.npy"),
    64: ("x_test_makeblobs.npy", "y_test_makeblobs.npy"),
}


def get_dataset(
        name: str,
        input_width: int,
        n_samples: int = 500,
        n_classes: int = 10,
        seed: int = 0,
        device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads one of the two server-hosted demo datasets, flattened/generated to
    match input_width. Both victim and clone must be compared on identical
    images, so this is called once and the same tensors are reused for both.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "make_blobs":
        return _get_make_blobs(input_width, n_samples, n_classes, seed, device)
    elif name == "cifar":
        return _get_cifar(input_width, n_samples, device)
    else:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {SUPPORTED_DATASETS}")


def _get_make_blobs(input_width, n_samples, n_classes, seed, device):
    if input_width in _MAKEBLOBS_TESTDATA_BY_WIDTH:
        X, y = _load_real_makeblobs_testdata(input_width, n_samples, seed)
    else:
        X, y = _make_blobs(n_samples=n_samples, n_features=input_width, centers=n_classes, random_state=seed)

    images = torch.from_numpy(X).float().to(device)
    labels = torch.from_numpy(y).long().to(device)
    return images, labels


def _load_real_makeblobs_testdata(input_width, n_samples, seed):
    x_name, y_name = _MAKEBLOBS_TESTDATA_BY_WIDTH[input_width]
    X = np.load(os.path.join(_MAKEBLOBS_TESTDATA_DIR, x_name))
    y = np.load(os.path.join(_MAKEBLOBS_TESTDATA_DIR, y_name))

    n = min(n_samples, len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def _get_cifar(input_width, n_samples, device):
    if input_width != CIFAR_FLATTENED_WIDTH:
        raise ValueError(
            f"cifar dataset is flattened 32x32x3 images ({CIFAR_FLATTENED_WIDTH} features) -- "
            f"architecture's first width must be {CIFAR_FLATTENED_WIDTH}, got {input_width}"
        )

    if not os.path.isdir(_CIFAR_LOCAL_DATASET_DIR):
        raise ValueError(
            f"Expected a real CIFAR-10 dataset at {_CIFAR_LOCAL_DATASET_DIR}, but that path doesn't exist. "
            f"This dataset is read from a fixed local path, not downloaded."
        )

    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=_CIFAR_LOCAL_DATASET_DIR, train=False, download=False, transform=transforms.ToTensor())
    n = min(n_samples, len(dataset))
    images = torch.stack([dataset[i][0] for i in range(n)]).reshape(n, -1).float().to(device)
    labels = torch.tensor([dataset[i][1] for i in range(n)]).long().to(device)
    return images, labels
