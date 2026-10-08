"""Unit tests for signal processing, normalization, and AAMI mapping."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("wfdb")

import numpy as np

from src.data.preprocess import (
    AAMI_CLASS_TO_INT,
    AAMI_MAPPING,
    apply_bandpass_filter,
    apply_zscore_norm,
)


def test_aami_mapping_coverage() -> None:
    """Ensures primary MIT-BIH symbols map into one of the 5 AAMI superclasses."""
    for symbol in ["N", "L", "R", "e", "j"]:
        assert AAMI_MAPPING[symbol] == "N"

    for symbol in ["A", "a", "J", "S"]:
        assert AAMI_MAPPING[symbol] == "S"

    for symbol in ["V", "E"]:
        assert AAMI_MAPPING[symbol] == "V"

    assert AAMI_MAPPING["F"] == "F"

    for symbol in ["/", "f", "Q"]:
        assert AAMI_MAPPING[symbol] == "Q"


def test_aami_class_to_int() -> None:
    """Ensures class to index mapping produces contiguous integers 0 to 4."""
    assert set(AAMI_CLASS_TO_INT.values()) == {0, 1, 2, 3, 4}


def test_zscore_normalization() -> None:
    """Verifies that Z-score normalization yields zero mean and unit variance."""
    raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    norm = apply_zscore_norm(raw)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)


def test_bandpass_filter_shape() -> None:
    """Verifies that the Butterworth filter preserves signal length."""
    t = np.linspace(0, 1, 360, endpoint=False)
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
    filtered = apply_bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=360.0)
    assert filtered.shape == (360,)
    assert filtered.dtype == np.float32
