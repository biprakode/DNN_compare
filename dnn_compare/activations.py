import numpy as np
import torch

from dnn_compare.models import NN

@torch.no_grad()
def extract_everything(model: NN, images) -> dict[str, np.ndarray]:
    output_layer_map, preds = model(images, return_all_layers = True)
    output_layer_map["preds"] = preds
    return {key: tensor.cpu().numpy() for key, tensor in output_layer_map.items()}