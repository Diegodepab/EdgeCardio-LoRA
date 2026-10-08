"""FastAPI service for real-time edge ECG arrhythmia classification.

Optimized for microcontrollers and embedded Linux gateways:
- Lifespan management for one-time ONNX engine loading.
- Zero PyTorch or PEFT imports in production runtime.
- Pydantic v2 data contracts preventing corrupted hardware telemetry.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.engine import ONNXEngine
from src.api.schemas import ECGRequest, ECGResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edgecardio_api")

# Global engine singleton reference
engine: ONNXEngine | None = None


def resolve_model_path() -> Path:
    """Resolves quantized ONNX model path from environment variable or project root."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "models" / "quantized" / "baseline_int8.onnx"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application lifecycle, loading ONNX engine once on cold start."""
    global engine
    model_path = resolve_model_path()
    logger.info("Initializing EdgeCardio-LoRA API service...")
    logger.info("Target model path: %s", model_path)

    try:
        engine = ONNXEngine(model_path)
    except Exception as exc:
        logger.error("Failed to load ONNX model on startup: %s", exc, exc_info=True)
        # Service can still boot to report unhealthy state on /health
        engine = None

    yield

    logger.info("Shutting down EdgeCardio-LoRA API service.")
    engine = None


app = FastAPI(
    title="EdgeCardio-LoRA Inference API",
    description="Low-latency Edge AI service for ECG arrhythmia classification using INT8 ONNX Runtime.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK, summary="Service health status")
async def health_check() -> JSONResponse:
    """Verifies service readiness and model availability."""
    is_ready = engine is not None
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_ready else "unhealthy",
            "model_loaded": is_ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "EdgeCardio-LoRA",
            "version": "0.1.0",
        },
    )


@app.post(
    "/v1/infer/beat",
    response_model=ECGResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify single ECG heartbeat",
)
async def infer_beat(request: ECGRequest) -> ECGResponse:
    """Classifies a 360-sample heartbeat window according to AAMI EC57 standard."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not loaded or unavailable. Check server logs.",
        )

    try:
        pred_class, class_name, confidence, inference_ms = engine.predict(request.signal)
    except Exception as exc:
        logger.error(
            "Inference execution failed for patient '%s': %s",
            request.patient_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(exc)}",
        ) from exc

    return ECGResponse(
        prediction=pred_class,
        class_name=class_name,
        confidence=round(confidence, 4),
        inference_ms=round(inference_ms, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
