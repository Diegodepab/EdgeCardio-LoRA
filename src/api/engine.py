"""High-performance ONNX Runtime inference engine for ECG classification.

Isolated from deep learning training frameworks: strictly uses onnxruntime
and numpy without any dependencies on PyTorch or PEFT.
"""

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger("onnx_engine")

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

AAMI_DESCRIPTIONS = {
    "N": "Normal / Bundle Branch Block",
    "S": "Supraventricular Ectopic Beat (SVEB)",
    "V": "Ventricular Ectopic Beat (VEB)",
    "F": "Fusion of Ventricular and Normal Beat",
    "Q": "Paced or Unclassifiable Beat",
}


class ONNXEngine:
    """Manages an ONNX Runtime InferenceSession for ultra-low latency prediction."""

    def __init__(self, model_path: Path | str) -> None:
        """Initializes and loads the quantized ONNX model once into memory.

        Args:
            model_path: Absolute or relative path to the baseline_int8.onnx model.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Quantized ONNX model not found at: {self.model_path}. "
                "Ensure Phase D quantization has been executed."
            )

        logger.info("Initializing ONNX Runtime session: %s", self.model_path)

        # Configure CPU session options optimized for Edge microservices
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1  # 1 thread minimizes core contention on edge CPUs
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warmup pass to prime CPU cache and instruction pipelines
        dummy_input = np.zeros((1, 1, 360), dtype=np.float32)
        _ = self.session.run([self.output_name], {self.input_name: dummy_input})
        logger.info("ONNX Runtime engine successfully warmed up and ready for inference.")

    @staticmethod
    def _normalize_zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Applies fast in-place Z-score normalization."""
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + eps)

    def predict(self, raw_signal: list[float]) -> tuple[str, str, float, float]:
        """Runs end-to-end inference on a single 360-sample heartbeat.

        Args:
            raw_signal: Validated list of 360 float values.

        Returns:
            Tuple containing:
            - prediction: AAMI class abbreviation ('N', 'S', 'V', 'F', 'Q')
            - class_name: Clinical description
            - confidence: Softmax confidence score (0.0 to 1.0)
            - inference_ms: Elapsed execution time in milliseconds
        """
        start_time = time.perf_counter()

        # 1. Reshape to (1, 1, 360) and normalize
        arr = np.array(raw_signal, dtype=np.float32).reshape(1, 1, 360)
        norm_arr = self._normalize_zscore(arr)

        # 2. Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: norm_arr})
        logits = outputs[0][0]  # shape: (5,)

        # 3. Compute numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # 4. Extract argmax class and confidence
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        pred_class = AAMI_CLASSES[class_idx]
        class_desc = AAMI_DESCRIPTIONS[pred_class]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return pred_class, class_desc, confidence, elapsed_ms
