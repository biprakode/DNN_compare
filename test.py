import torch
from dnn_compare.models import NN
from dnn_compare.activations import extract_everything

model = NN([4, 8, 8, 3])       # pick whatever widths mirror your real checkpoints
images = torch.randn(5, 4)     # 5 dummy samples, input width 4

out = extract_everything(model, images)
for k, v in out.items():
    print(k, v.shape, v.dtype)