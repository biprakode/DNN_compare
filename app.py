"""FastAPI app exposing the single unified /compare endpoint the frontend
prototype (DNN Model Comparison Tool-handoff/) calls, plus serving that
prototype as static files for the demo.

See HANDOFF.md for the reconciled request/response contract.
"""
import json
import queue
import secrets
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dnn_compare.datasets import CIFAR_FLATTENED_WIDTH, SUPPORTED_DATASETS, get_dataset
from dnn_compare.io import load_checkpoint_bytes
from dnn_compare.report import compute_full_report, compute_full_report_with_progress
from knobs import apply_layer_knobs

STATIC_DIR = "DNN Model Comparison Tool-handoff/dnn-model-comparison-tool/project"
ACTIVATIONS = ("relu", "leaky_relu")
CLONE_MODES = ("copy", "upload")


class APIError(Exception):
    def __init__(self, status_code: int, code: str, detail: str):
        self.status_code = status_code
        self.code = code
        self.detail = detail


app = FastAPI(title="DNN model comparison tool API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")


@app.get("/")
def root():
    return RedirectResponse(url="/ui/DNN Comparison Tool.dc.html")


@app.exception_handler(APIError)
def handle_api_error(request: Request, exc: APIError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "detail": exc.detail})


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


def _parse_architecture(raw: str, field_name: str) -> list[int]:
    widths = [int(s.strip()) for s in raw.split(",") if s.strip().lstrip("-").isdigit()]
    if len(widths) < 2 or any(w <= 0 for w in widths):
        raise APIError(422, "invalid_architecture", f"{field_name} must be at least two positive comma-separated widths, got {raw!r}")
    return widths


def _validate_activation(value: str, field_name: str) -> None:
    if value not in ACTIVATIONS:
        raise APIError(422, "invalid_activation", f"{field_name} must be one of {ACTIVATIONS}, got {value!r}")


def _parse_knobs(raw: str, n_layers: int, layer_widths: list[int]) -> list[dict]:
    try:
        knobs = json.loads(raw)
    except json.JSONDecodeError as e:
        raise APIError(422, "invalid_knobs", f"knobs must be valid JSON: {e}") from e

    if not isinstance(knobs, list) or len(knobs) != n_layers:
        raise APIError(422, "invalid_knobs", f"knobs must be a list of {n_layers} per-layer objects, got {knobs!r}")

    for i, k in enumerate(knobs):
        out_features = layer_widths[i]
        for field in ("signature", "cosine", "sign"):
            if field not in k:
                raise APIError(422, "invalid_knobs", f"knobs[{i}] missing required field {field!r}")
        if not (0 <= k["signature"] <= out_features):
            raise APIError(422, "invalid_knobs", f"knobs[{i}].signature={k['signature']} out of range for {out_features} neurons")
        if not (0.0 <= k["cosine"] <= 1.0):
            raise APIError(422, "invalid_knobs", f"knobs[{i}].cosine={k['cosine']} must be in [0, 1]")
        if not (0 <= k["sign"] <= out_features):
            raise APIError(422, "invalid_knobs", f"knobs[{i}].sign={k['sign']} out of range for {out_features} neurons")

    return knobs


def _build_tuned_clone_bytes(victim_state_dict: dict, widths: list[int], knobs: list[dict]) -> dict:
    """Applies per-layer knobs to a copy of the victim's state_dict's weight
    matrices (biases are untouched by all three knobs), returning a new
    state_dict for the clone model.
    """
    base_seed = secrets.randbits(32)
    mutated_state_dict = {k: v.clone() for k, v in victim_state_dict.items()}

    for i in range(len(widths) - 1):
        weight_key = f"layer_map.layer_{i}.weight"
        W = mutated_state_dict[weight_key].cpu().numpy()

        mutated_W = apply_layer_knobs(
            W,
            unrecovered_count=knobs[i]["signature"],
            target_cosine=knobs[i]["cosine"],
            sign_flip_count=knobs[i]["sign"],
            signature_seed=base_seed + i * 2,
            sign_seed=base_seed + i * 2 + 1,
        )
        mutated_state_dict[weight_key] = torch.from_numpy(mutated_W).float()

    return mutated_state_dict


def _validate_and_load_models(
        victim_checkpoint: UploadFile,
        victim_architecture: str,
        victim_activation: str,
        dataset: str,
        clone_mode: str,
        knobs_json: str | None,
        clone_checkpoint: UploadFile | None,
        clone_architecture: str | None,
        clone_activation: str | None,
):
    """Validates every field (fail fast, before any model loading, dataset
    loading, or computation) and returns (victim_model, clone_model,
    victim_widths, dataset). Shared by both /compare and /compare/stream so
    the two endpoints can't drift apart on validation or model-building
    behavior. Dataset loading itself is NOT done here -- for CIFAR that can
    involve a slow one-time download, and /compare/stream needs to report
    progress on that separately from model loading.
    """
    # --- validation first, before any model loading or computation ---
    victim_widths = _parse_architecture(victim_architecture, "victim_architecture")
    _validate_activation(victim_activation, "victim_activation")

    if dataset not in SUPPORTED_DATASETS:
        raise APIError(422, "invalid_dataset", f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}")
    if dataset == "cifar" and victim_widths[0] != CIFAR_FLATTENED_WIDTH:
        raise APIError(422, "invalid_architecture", f"cifar dataset requires architecture's first width to be {CIFAR_FLATTENED_WIDTH}, got {victim_widths[0]}")

    if clone_mode not in CLONE_MODES:
        raise APIError(422, "invalid_clone_mode", f"clone_mode must be one of {CLONE_MODES}, got {clone_mode!r}")

    knobs = None
    if clone_mode == "copy":
        if knobs_json is None:
            raise APIError(422, "missing_knobs", "clone_mode=copy requires knobs_json")
        knobs = _parse_knobs(knobs_json, n_layers=len(victim_widths) - 1, layer_widths=victim_widths[1:])
        clone_widths = victim_widths
    else:
        if clone_checkpoint is None or clone_architecture is None:
            raise APIError(422, "missing_clone_upload", "clone_mode=upload requires clone_checkpoint and clone_architecture")
        clone_activation_value = clone_activation or "relu"
        _validate_activation(clone_activation_value, "clone_activation")
        clone_widths = _parse_architecture(clone_architecture, "clone_architecture")
        if clone_widths != victim_widths:
            raise APIError(422, "architecture_mismatch", f"uploaded clone architecture {clone_widths} must match victim architecture {victim_widths} -- every metric requires identical layer shapes")

    # --- load victim, build clone ---
    # Sync (not async) route: FastAPI runs this in a worker thread, so blocking
    # I/O here (checkpoint reads, knob math) never stalls the event loop for
    # other in-flight requests.
    victim_bytes = victim_checkpoint.file.read()
    victim_model = load_checkpoint_bytes(victim_bytes, victim_widths, victim_activation)

    if clone_mode == "copy":
        mutated_state_dict = _build_tuned_clone_bytes(victim_model.state_dict(), victim_widths, knobs)
        from dnn_compare.models import NN
        clone_model = NN(victim_widths, victim_activation)
        clone_model.load_state_dict(mutated_state_dict)
        clone_model.eval()
    else:
        clone_bytes = clone_checkpoint.file.read()
        clone_model = load_checkpoint_bytes(clone_bytes, clone_widths, clone_activation_value)

    return victim_model, clone_model, victim_widths


@app.post("/compare")
def compare(
        victim_checkpoint: UploadFile,
        victim_architecture: str = Form(...),
        victim_activation: str = Form("relu"),
        dataset: str = Form(...),
        clone_mode: str = Form(...),
        knobs_json: str | None = Form(None),
        clone_checkpoint: UploadFile | None = None,
        clone_architecture: str | None = Form(None),
        clone_activation: str | None = Form(None),
):
    victim_model, clone_model, victim_widths = _validate_and_load_models(
        victim_checkpoint, victim_architecture, victim_activation, dataset, clone_mode,
        knobs_json, clone_checkpoint, clone_architecture, clone_activation,
    )
    n_samples = 60
    images, labels = get_dataset(dataset, input_width=victim_widths[0], n_samples=n_samples)
    report = compute_full_report(victim_model, clone_model, images, labels.cpu().numpy())
    return report


@app.post("/compare/stream")
def compare_stream(
        victim_checkpoint: UploadFile,
        victim_architecture: str = Form(...),
        victim_activation: str = Form("relu"),
        dataset: str = Form(...),
        clone_mode: str = Form(...),
        knobs_json: str | None = Form(None),
        clone_checkpoint: UploadFile | None = None,
        clone_architecture: str | None = Form(None),
        clone_activation: str | None = Form(None),
):
    """Same comparison as /compare, but streams newline-delimited JSON
    progress events while it runs instead of returning one final response:
    {"type": "progress", "tested": int, "total": int, "eta_seconds": float|null}
    lines while the forward passes are chunked through, then either
    {"type": "done", "report": {...}} or {"type": "error", "error": str, "detail": str}.
    Validation errors are still raised as a normal APIError/422 response --
    they happen before the stream starts, same as /compare. Dataset loading
    (make_blobs generation, or CIFAR read from its fixed local path) is fast
    enough that it doesn't get its own progress phase.
    """
    victim_model, clone_model, victim_widths = _validate_and_load_models(
        victim_checkpoint, victim_architecture, victim_activation, dataset, clone_mode,
        knobs_json, clone_checkpoint, clone_architecture, clone_activation,
    )
    n_samples = 60

    def event_stream():
        events: queue.Queue = queue.Queue()
        start_time = time.time()

        def on_progress(tested, total):
            events.put({"kind": "progress", "tested": tested, "total": total})

        def worker():
            try:
                images, labels = get_dataset(dataset, input_width=victim_widths[0], n_samples=n_samples)
                labels_np = labels.cpu().numpy()
                report = compute_full_report_with_progress(victim_model, clone_model, images, labels_np, on_progress=on_progress)
                events.put({"kind": "done", "report": report})
            except Exception as e:
                events.put({"kind": "error", "detail": str(e)})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            event = events.get()
            kind = event["kind"]

            if kind == "progress":
                tested, total = event["tested"], event["total"]
                elapsed = time.time() - start_time
                rate = tested / elapsed if elapsed > 0 else 0
                eta_seconds = (total - tested) / rate if rate > 0 else None
                yield json.dumps({"type": "progress", "tested": tested, "total": total, "eta_seconds": eta_seconds}) + "\n"
            elif kind == "done":
                yield json.dumps({"type": "done", "report": event["report"]}) + "\n"
                return
            else:
                yield json.dumps({"type": "error", "error": "internal_error", "detail": event["detail"]}) + "\n"
                return

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
