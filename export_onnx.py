"""
RTIE — ONNX Export & Inference Benchmark

Exports the trained model to ONNX format and benchmarks
PyTorch vs ONNX inference latency.
"""

import os
import json
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

import config
from model import RTIEModel


class RTIEONNXWrapper(torch.nn.Module):
    """Wrapper that takes concatenated input for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, physics_features):
        out = self.model(images, physics_features)
        return out["logits"], out["efficiency"], out["confidence"]


def export_to_onnx():
    """Export model to ONNX format."""
    device = torch.device("cpu")  # ONNX export on CPU
    model = RTIEModel(pretrained=False).to(device)

    ckpt_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapper = RTIEONNXWrapper(model)
    wrapper.eval()

    # Dummy inputs
    dummy_images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    dummy_physics = torch.randn(1, config.NUM_PHYSICS_FEATURES)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(config.MODEL_DIR, "rtie_model.onnx")

    # Turn off MC Dropout for deterministic ONNX export
    for module in wrapper.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0  # disable dropout for export

    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_physics),
        onnx_path,
        input_names=["images", "physics_features"],
        output_names=["logits", "efficiency", "confidence"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "physics_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "efficiency": {0: "batch_size"},
            "confidence": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {onnx_path}")
    print(f"  Model size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    return onnx_path


def benchmark_inference(onnx_path, num_iterations=100):
    """Benchmark PyTorch vs ONNX inference latency."""
    device = torch.device("cpu")

    # ── PyTorch Benchmark ──
    model = RTIEModel(pretrained=False).to(device)
    ckpt = torch.load(os.path.join(config.MODEL_DIR, "best_model.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_img = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    dummy_phys = torch.randn(1, config.NUM_PHYSICS_FEATURES)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_img, dummy_phys)

    pytorch_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_img, dummy_phys)
        pytorch_times.append((time.perf_counter() - start) * 1000)

    # ── ONNX Benchmark ──
    session = ort.InferenceSession(onnx_path)

    img_np = dummy_img.numpy()
    phys_np = dummy_phys.numpy()

    # Warmup
    for _ in range(10):
        session.run(None, {"images": img_np, "physics_features": phys_np})

    onnx_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {"images": img_np, "physics_features": phys_np})
        onnx_times.append((time.perf_counter() - start) * 1000)

    results = {
        "pytorch": {
            "mean_ms": round(np.mean(pytorch_times), 2),
            "std_ms": round(np.std(pytorch_times), 2),
            "p50_ms": round(np.percentile(pytorch_times, 50), 2),
            "p95_ms": round(np.percentile(pytorch_times, 95), 2),
        },
        "onnx": {
            "mean_ms": round(np.mean(onnx_times), 2),
            "std_ms": round(np.std(onnx_times), 2),
            "p50_ms": round(np.percentile(onnx_times, 50), 2),
            "p95_ms": round(np.percentile(onnx_times, 95), 2),
        },
        "speedup": round(np.mean(pytorch_times) / np.mean(onnx_times), 2),
        "num_iterations": num_iterations,
    }

    print(f"\nInference Benchmark ({num_iterations} iterations):")
    print(f"  PyTorch — Mean: {results['pytorch']['mean_ms']:.2f} ms | P95: {results['pytorch']['p95_ms']:.2f} ms")
    print(f"  ONNX    — Mean: {results['onnx']['mean_ms']:.2f} ms | P95: {results['onnx']['p95_ms']:.2f} ms")
    print(f"  Speedup: {results['speedup']:.2f}×")

    # Save
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    json_path = os.path.join(config.REPORT_DIR, "benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results → {json_path}")

    return results


if __name__ == "__main__":
    onnx_path = export_to_onnx()
    benchmark_inference(onnx_path)
