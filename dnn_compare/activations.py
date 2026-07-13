import numpy as np
import torch

from dnn_compare.models import NN

@torch.no_grad()
def extract_all_layer_activations(model: NN, images) -> dict[str, np.ndarray]:
    output_layer_map, _ = model(images, return_all_layers = True)
    return {key: tensor.cpu().numpy() for key, tensor in output_layer_map.items()}

@torch.no_grad()
def extract_predictions(model: NN, images) -> np.ndarray:
    _, preds = model(images, return_all_layers = False)
    return preds.cpu().numpy()