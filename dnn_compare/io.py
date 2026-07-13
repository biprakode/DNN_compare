import io as _io
import torch
import numpy as np
import os

from dnn_compare.models import NN


def _remap_fc_style_state_dict(state_dict: dict, n_layers: int) -> dict | None:
    """Checkpoints trained outside this repo (e.g. the paper's reference
    victims in tiny_stuff/) use a plain "fc1, fc2, ..." nn.Linear naming
    convention instead of this repo's "layer_map.layer_0, layer_map.layer_1,
    ..." ModuleDict naming. Same architecture, different key names. Returns a
    remapped state_dict if `state_dict` matches that fc{i}.{weight,bias}
    pattern for exactly n_layers layers, else None.
    """
    expected_keys = {f"fc{i + 1}.weight" for i in range(n_layers)} | {f"fc{i + 1}.bias" for i in range(n_layers)}
    if set(state_dict.keys()) != expected_keys:
        return None

    remapped = {}
    for i in range(n_layers):
        remapped[f"layer_map.layer_{i}.weight"] = state_dict[f"fc{i + 1}.weight"]
        remapped[f"layer_map.layer_{i}.bias"] = state_dict[f"fc{i + 1}.bias"]
    return remapped


def _build_model_from_state_dict(state_dict, widths: list[int], activation: str, device) -> NN:
    model = NN(widths, activation)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as original_error:
        remapped = _remap_fc_style_state_dict(state_dict, n_layers=len(widths) - 1)
        if remapped is not None:
            model.load_state_dict(remapped)
        else:
            raise ValueError(
                f"Checkpoint state_dict does not match widths={widths}, activation={activation!r}: {original_error}"
            ) from original_error

    model.to(device)
    model.eval()
    return model


def load_checkpoint(path: str, widths: list[int], activation: str = "relu", device=None) -> NN:
    """
    Instantiates the model, loads state_dict from path,
    sets to eval mode, and moves to device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # map_location ensures compatibility if moving between GPU/CPU
    state_dict = torch.load(path, map_location=device, weights_only=True)
    return _build_model_from_state_dict(state_dict, widths, activation, device)


def load_checkpoint_bytes(file_bytes: bytes, widths: list[int], activation: str = "relu", device=None) -> NN:
    """
    Same as load_checkpoint, but reads a state_dict from raw bytes (e.g. an
    uploaded file's content) instead of a filesystem path -- lets the API
    layer load an UploadFile without ever writing it to disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state_dict = torch.load(_io.BytesIO(file_bytes), map_location=device, weights_only=True)
    except Exception as e:
        raise ValueError(f"Could not parse checkpoint bytes as a torch state_dict: {e}") from e

    return _build_model_from_state_dict(state_dict, widths, activation, device)

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