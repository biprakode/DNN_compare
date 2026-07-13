import torch
from torch import nn


class NN(nn.Module):
    def __init__(self , widths : list[int] , activation : str = "relu"):
        super(NN, self).__init__()
        self.n_widths = len(widths)
        self.layer_map = nn.ModuleDict()

        for i in range(len(widths) - 1):
            self.layer_map[f"layer_{i}"] = nn.Linear(widths[i], widths[i + 1])
            if i < len(widths) - 2:
                if activation == "relu":
                    self.layer_map[f"non_linearity_{i}"] = nn.ReLU()
                elif activation == "leaky_relu":
                    self.layer_map[f"non_linearity_{i}"] = nn.LeakyReLU()
                else:
                    raise ValueError(f"Unrecognized activation: {activation!r}")

        self.layer_map["softmax"] = nn.Softmax(dim=1)

    def forward(self, x , return_all_layers = False):
        output_layer_map = {}
        if return_all_layers:
            out = x
            for i in range(self.n_widths - 1):
                if i < self.n_widths - 2:
                    out = self.layer_map[f"non_linearity_{i}"](self.layer_map[f"layer_{i}"](out))
                    output_layer_map[f"act_{i}"] = out
                else:
                    out = self.layer_map[f"layer_{i}"](out)
                    output_layer_map["output"] = out
            return output_layer_map , torch.argmax(output_layer_map["output"], dim = 1)
        else:
            out = x
            for module in self.layer_map.values():
                out = module(out)
            probs = out
            preds = torch.argmax(probs, dim = 1)
            return probs , preds