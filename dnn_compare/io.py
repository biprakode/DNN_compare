import torch
import numpy as np
import os

from dnn_compare.models import NN


def load_checkpoint(path: str, widths: list[int], activation: str = "relu", device=None) -> NN:
    """
    Instantiates the model, loads state_dict from path,
    sets to eval mode, and moves to device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate
    model = NN(widths, activation)

    # 2. Load state dict
    # map_location ensures compatibility if moving between GPU/CPU
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Finalize
    model.to(device)
    model.eval()

    return model

def load_dataset(path: str, expected_input_width: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads .npz file containing 'images' and 'labels'.
    Validates input width before returning tensors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Load the .npz file
    data = np.load(path)
    images = data['images']
    labels = data['labels']

    # Sanity check for dimensions
    if images.shape[1] != expected_input_width:
        raise ValueError(
            f"Input dimension mismatch: Dataset images have shape {images.shape[1]}, "
            f"but model expects {expected_input_width}."
        )

    # Convert to Tensors
    # Assuming standard float32 for images and int64 for labels
    images_tensor = torch.from_numpy(images).float().to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    return images_tensor, labels_tensor