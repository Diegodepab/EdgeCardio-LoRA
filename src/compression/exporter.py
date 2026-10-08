"""ONNX export module with LoRA weight absorption (merge_and_unload).

Merges patient-specific LoRA adapters directly into the baseline ECG1DCNN
weights and exports the graph to an optimized ONNX model, removing any
runtime dependency on PyTorch or PEFT.
"""

import argparse
import logging
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from peft import PeftModel

from src.adaptation.lora_peft import load_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("onnx_exporter")


def export_merged_to_onnx(
    baseline_weights_path: Path | str,
    adapter_path: Path | str | None = None,
    output_path: Path | str | None = None,
    opset_version: int = 14,
) -> Path:
    """Loads base model, merges LoRA adapter weights, and exports to ONNX.

    Args:
        baseline_weights_path: Path to baseline.pt pre-trained checkpoint.
        adapter_path: Path to directory containing LoRA adapter weights
                      (e.g. models/lora_adapters/patient_01).
        output_path: Path for output .onnx file
                     (default: models/quantized/baseline_merged.onnx).
        opset_version: ONNX operator set version (default: 14).

    Returns:
        Path to the exported ONNX file.
    """
    repo_root = Path(__file__).resolve().parents[2]
    baseline_weights = Path(baseline_weights_path)

    if output_path is None:
        output_path = repo_root / "models" / "quantized" / "baseline_merged.onnx"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load base PyTorch model
    logger.info("Instantiating base ECG1DCNN model...")
    base_model = load_baseline_model(weights_path=baseline_weights)
    base_model.eval()

    # 2. Merge LoRA adapter weights if provided
    if adapter_path is not None and Path(adapter_path).is_dir():
        adapter_dir = Path(adapter_path)
        logger.info("Applying LoRA adapter from %s...", adapter_dir)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        logger.info("Fusing LoRA weights into base model (merge_and_unload)...")
        export_model: nn.Module = peft_model.merge_and_unload()
    else:
        logger.info(
            "No LoRA adapter specified or directory not found. Exporting base model directly."
        )
        export_model = base_model

    export_model.eval()

    # 3. Create dummy input tensor: (batch_size=1, channels=1, sequence_length=360)
    dummy_input = torch.randn(1, 1, 360, dtype=torch.float32)

    # 4. Export graph to ONNX format
    logger.info("Exporting fused model to ONNX: %s (opset %d)...", output_path, opset_version)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ecg"],
        output_names=["logits"],
        dynamic_axes={
            "input_ecg": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # 5. Verify exported ONNX model integrity
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    file_size_kb = output_path.stat().st_size / 1024.0

    logger.info("=" * 65)
    logger.info("ONNX export successfully validated!")
    logger.info("Output file: %s", output_path)
    logger.info("File size: %.2f KB (%.3f MB)", file_size_kb, file_size_kb / 1024.0)
    logger.info("Dynamic batch size enabled: shape (batch_size, 1, 360)")
    logger.info("=" * 65)

    return output_path


def main() -> None:
    """CLI entrypoint for ONNX model export."""
    repo_root = Path(__file__).resolve().parents[2]
    default_base_weights = repo_root / "models" / "baseline" / "baseline.pt"
    default_adapter = repo_root / "models" / "lora_adapters" / "patient_01"
    default_output = repo_root / "models" / "quantized" / "baseline_merged.onnx"

    parser = argparse.ArgumentParser(description="Merge LoRA and export ECG1DCNN to ONNX")
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=str(default_base_weights),
        help=f"Path to baseline.pt (default: {default_base_weights})",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=str(default_adapter),
        help=f"Path to LoRA adapter folder (default: {default_adapter})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output ONNX file (default: {default_output})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    export_merged_to_onnx(
        baseline_weights_path=args.baseline_weights,
        adapter_path=args.adapter_dir,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
