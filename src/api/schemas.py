"""Pydantic v2 schemas for ECG edge inference validation and response contracts."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ECGRequest(BaseModel):
    """Validation schema for incoming ECG beat signal payload from edge sensors."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "patient_01",
                "signal": [0.0] * 360,
            }
        }
    )

    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique patient identifier (e.g. 'patient_01', 'P-203')",
    )
    signal: list[float] = Field(
        ...,
        description="Discrete Lead II ECG beat array consisting of exactly 360 numerical samples",
    )

    @field_validator("signal")
    @classmethod
    def validate_signal_length_and_values(cls, v: list[float]) -> list[float]:
        """Ensures incoming payload is exactly 360 finite numbers without NaNs or Infs."""
        if len(v) != 360:
            raise ValueError(
                f"Invalid signal dimensions: expected exactly 360 samples, received {len(v)}."
            )

        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Corrupted sensor data: sample at index {i} is not a finite number (NaN/Inf)."
                )

        return v


class ECGResponse(BaseModel):
    """Response contract for classified ECG heartbeat."""

    prediction: str = Field(
        ...,
        description="Classified AAMI EC57 superclass: 'N', 'S', 'V', 'F', or 'Q'",
        example="N",
    )
    class_name: str = Field(
        ...,
        description="Clinical nomenclature of the detected rhythm category",
        example="Normal / Bundle Branch Block",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized softmax confidence score in range [0.0, 1.0]",
        example=0.9842,
    )
    inference_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference execution time in milliseconds",
        example=0.28,
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of prediction generation",
        example="2026-10-08T12:00:00.000000Z",
    )
