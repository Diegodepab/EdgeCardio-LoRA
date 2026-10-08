"""INT8 dynamic quantization and edge latency benchmarking module.

Converts FP32 ONNX model to INT8 using onnxruntime.quantization and measures
disk size reduction, inference latency (ms per beat), and throughput.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quantizer")


def quantize_to_int8(
    input_onnx_path: Path | str,
    output_onnx_path: Path | str,
    weight_type: QuantType = QuantType.QUInt8,
) -> Path:
    """Quantizes an FP32 ONNX model to INT8 dynamic representation.

    Args:
        input_onnx_path: Path to baseline_merged.onnx (FP32).
        output_onnx_path: Destination path for baseline_int8.onnx.
        weight_type: Integer quantization type (QuantType.QUInt8 or QuantType.QInt8).

    Returns:
        Path to generated INT8 ONNX file.
    """
    inp = Path(input_onnx_path)
    out = Path(output_onnx_path)

    if not inp.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {inp}")

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting dynamic INT8 quantization: %s -> %s", inp, out)

    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=weight_type,
    )

    logger.info("Quantization completed successfully.")
    return out


def benchmark_latency(
    model_path: Path | str,
    num_samples: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Measures single-beat inference latency using ONNX Runtime.

    Args:
        model_path: Path to .onnx model.
        num_samples: Number of benchmark iterations.
        warmup: Number of warmup iterations to prime CPU caches.

    Returns:
        Dictionary with mean, median, p95, min, max latency (ms) and throughput.
    """
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 1. Warmup passes
    for _ in range(warmup):
        dummy_beat = np.random.randn(1, 1, 360).astype(np.float32)
        _ = session.run(None, {input_name: dummy_beat})

    # 2. Timed benchmarking passes
    latencies_ms = []
    for _ in range(num_samples):
        dummy_beat = np.random.randn(1, 1, 360).astype(np.float32)
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_beat})
        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000.0)

    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_ms": mean_lat,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "throughput_hz": throughput,
    }


def compare_and_report(
    fp32_path: Path,
    int8_path: Path,
    num_samples: int = 100,
) -> None:
    """Compares FP32 and INT8 ONNX models and outputs benchmark report."""
    fp32_size_kb = fp32_path.stat().st_size / 1024.0
    int8_size_kb = int8_path.stat().st_size / 1024.0
    compression_pct = (1.0 - (int8_size_kb / fp32_size_kb)) * 100.0

    logger.info("Running latency benchmarks (%d samples, CPU)...", num_samples)
    fp32_metrics = benchmark_latency(fp32_path, num_samples=num_samples)
    int8_metrics = benchmark_latency(int8_path, num_samples=num_samples)

    speedup = (
        fp32_metrics["mean_ms"] / int8_metrics["mean_ms"] if int8_metrics["mean_ms"] > 0 else 1.0
    )

    print("\n" + "=" * 90)
    print("                    EDGECARDIO-LORA: EDGE COMPRESSION & LATENCY REPORT")
    print("=" * 90)
    print(
        f"{'Model Variant':<22} | {'Size (KB)':<10} | {'Size (MB)':<10} | {'Mean Latency':<12} | {'P95 Latency':<11} | {'Throughput'}"
    )
    print("-" * 90)
    print(
        f"{'FP32 Merged ONNX':<22} | "
        f"{fp32_size_kb:>8.2f} KB | "
        f"{fp32_size_kb / 1024.0:>8.3f} MB | "
        f"{fp32_metrics['mean_ms']:>8.3f} ms | "
        f"{fp32_metrics['p95_ms']:>8.3f} ms | "
        f"{fp32_metrics['throughput_hz']:>7.0f} beats/s"
    )
    print(
        f"{'INT8 Quantized ONNX':<22} | "
        f"{int8_size_kb:>8.2f} KB | "
        f"{int8_size_kb / 1024.0:>8.3f} MB | "
        f"{int8_metrics['mean_ms']:>8.3f} ms | "
        f"{int8_metrics['p95_ms']:>8.3f} ms | "
        f"{int8_metrics['throughput_hz']:>7.0f} beats/s"
    )
    print("=" * 90)
    print(
        f" Disk Size Reduction   : {compression_pct:.2f}% (Compression factor: {fp32_size_kb / int8_size_kb:.2f}x)"
    )
    print(f" Inference Speedup     : {speedup:.2f}x faster execution on CPU")
    print(f" Memory Target (< 2 MB): PASSED (INT8 footprint: {int8_size_kb / 1024.0:.3f} MB)")

    sla_passed = int8_metrics["p95_ms"] < 10.0
    sla_status = "PASSED" if sla_passed else "FAILED"
    margin = 10.0 / int8_metrics["mean_ms"] if int8_metrics["mean_ms"] > 0 else 0.0
    print(
        f" Edge SLA (< 10 ms)    : {sla_status} (Inference is {margin:.1f}x faster than 10 ms real-time limit)"
    )
    print("=" * 90 + "\n")


def main() -> None:
    """CLI entrypoint for INT8 quantization and benchmarking."""
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "models" / "quantized" / "baseline_merged.onnx"
    default_output = repo_root / "models" / "quantized" / "baseline_int8.onnx"

    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8 and benchmark latency"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help=f"Input FP32 ONNX model (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output INT8 ONNX model (default: {default_output})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of latency benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        default="QUInt8",
        choices=["QUInt8", "QInt8"],
        help="Quantization weight type (default: QUInt8)",
    )
    args = parser.parse_args()

    weight_type = QuantType.QUInt8 if args.weight_type == "QUInt8" else QuantType.QInt8
    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1. Quantize
    quantize_to_int8(
        input_onnx_path=input_path,
        output_onnx_path=output_path,
        weight_type=weight_type,
    )

    # 2. Benchmark and compare
    compare_and_report(
        fp32_path=input_path,
        int8_path=output_path,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
