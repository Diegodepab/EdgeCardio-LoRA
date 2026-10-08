"""Unit tests for FastAPI Pydantic v2 schemas and data validation."""

import pytest
from pydantic import ValidationError

from src.api.schemas import ECGRequest, ECGResponse


def test_ecg_request_valid() -> None:
    """Validates that a correctly shaped 360-sample array passes validation."""
    valid_signal = [0.12 * (i % 10) for i in range(360)]
    req = ECGRequest(patient_id="patient_01", signal=valid_signal)
    assert req.patient_id == "patient_01"
    assert len(req.signal) == 360


def test_ecg_request_invalid_length() -> None:
    """Validates that signals of length != 360 are rejected with ValidationError."""
    with pytest.raises(ValidationError, match="expected exactly 360 samples"):
        ECGRequest(patient_id="patient_01", signal=[0.0] * 359)

    with pytest.raises(ValidationError, match="expected exactly 360 samples"):
        ECGRequest(patient_id="patient_01", signal=[0.0] * 361)


def test_ecg_request_nan_or_inf_rejection() -> None:
    """Validates that corrupt sensor data containing NaN or Inf is rejected."""
    nan_signal = [0.0] * 359 + [float("nan")]
    with pytest.raises(ValidationError, match="not a finite number"):
        ECGRequest(patient_id="patient_01", signal=nan_signal)

    inf_signal = [0.0] * 359 + [float("inf")]
    with pytest.raises(ValidationError, match="not a finite number"):
        ECGRequest(patient_id="patient_01", signal=inf_signal)


def test_ecg_response_contract() -> None:
    """Validates response model field serialization and constraints."""
    res = ECGResponse(
        prediction="V",
        class_name="Ventricular Ectopic Beat (VEB)",
        confidence=0.9842,
        inference_ms=0.284,
        timestamp="2026-10-08T12:00:00Z",
    )
    assert res.prediction == "V"
    assert res.confidence == 0.9842
    assert res.inference_ms == 0.284

