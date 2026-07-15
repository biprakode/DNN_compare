# DNN_compare

`dnncompare` is a shared Python library used by two apps:

- **Project 1** compares any two model checkpoints on naive, representational,
  and functional similarity metrics.
- **Project 2** takes a single "victim" checkpoint, mutates it into a "clone"
  via per-layer knobs (signature reinit, cosine-similarity targeting, sign
  flipping), and reuses the exact same reporting pipeline to compare victim
  vs. clone.

Both apps funnel through one orchestrator, `dnn_compare/report.py`'s
`compute_full_report(model_a, model_b, images, labels)`, which returns a
`{"naive": ..., "representational": ..., "functional": ...}` dict.

## Running the live demo

```bash
python3 -m uvicorn app:app --port 8731
```

Then open `http://127.0.0.1:8731/` — it redirects to the real prototype UI
(served straight from the handoff bundle), which now calls the real backend
instead of faking data. Fill in the victim form, choose a clone mode, hit
Recalculate.

- `dataset=make_blobs` works instantly, any architecture.
- `dataset=cifar` requires an architecture whose first width is exactly 3072
  (32x32x3 flattened). Reads real CIFAR-10 from a hardcoded local path
  (`dnn_compare/datasets.py`'s `_CIFAR_LOCAL_DATASET_DIR`) instead of
  downloading — fast on this machine, but that path needs repointing if this
  ever runs somewhere else without CIFAR-10 already sitting on disk.

## Quickstart (library only, no server)

```bash
python3 test.py
```

Exercises `NN` + `extract_everything` end to end on a dummy 4-8-8-3 network
and prints each activation layer's shape/dtype. No real checkpoints or
datasets required.

The package lives under `dnn_compare/`. See `HANDOFF.md` for the API/data
contract, what's tested vs. not, and open questions.
