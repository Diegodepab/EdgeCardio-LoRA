"""Deterministic ECG preprocessing and segmentation module for MIT-BIH database.

Applies:
1. Lead II (MLII) channel extraction.
2. Butterworth bandpass filtering (0.5 Hz - 45 Hz) via filtfilt (zero-phase distortion).
3. Z-score normalization.
4. R-peak centered beat windowing (360 samples = 120 before + 240 after R-peak at 360 Hz).
5. AAMI EC57 5-superclass mapping (N, S, V, F, Q).
6. de Chazal inter-patient split (DS1 train/val, DS2 independent test).
"""

import logging
from pathlib import Path

import numpy as np
import scipy.signal
import wfdb

from src.data.ingest import DE_CHAZAL_DS1, DE_CHAZAL_DS2, EXCLUDED_PACEMAKER_RECORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# AAMI EC57 1998 standard mapping: PhysioNet annotation -> Superclass
AAMI_MAPPING: dict[str, str] = {
    # N: Normal or bundle branch block beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: Supraventricular ectopic beats (SVEB)
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: Ventricular ectopic beats (VEB)
    "V": "V",
    "E": "V",
    # F: Fusion beats
    "F": "F",
    # Q: Paced or unclassifiable beats
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

AAMI_CLASS_TO_INT: dict[str, int] = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}


def load_raw_lead_ii(record_path: Path) -> tuple[np.ndarray, float]:
    """Reads raw record and extracts Lead II (MLII) signal and sampling frequency."""
    record = wfdb.rdrecord(str(record_path))

    # Identify channel: prioritize MLII / II, default to channel 0
    channel_idx = 0
    if "MLII" in record.sig_name:
        channel_idx = record.sig_name.index("MLII")
    elif "II" in record.sig_name:
        channel_idx = record.sig_name.index("II")

    raw_signal = record.p_signal[:, channel_idx]
    cleaned_signal = np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned_signal.astype(np.float32), float(record.fs)


def apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 360.0,
    order: int = 3,
) -> np.ndarray:
    """Applies zero-phase Butterworth bandpass filter (0.5 Hz - 45 Hz).

    Uses filtfilt to avoid phase distortion and ensure R-peak temporal alignment.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype="bandpass")
    filtered = scipy.signal.filtfilt(b, a, signal)
    return np.asarray(filtered, dtype=np.float32)


def apply_zscore_norm(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Applies Z-score normalization: (signal - mean) / (std + eps)."""
    mean = np.mean(signal)
    std = np.std(signal)
    return np.asarray((signal - mean) / (std + eps), dtype=np.float32)


def segment_record_beats(
    raw_dir: Path,
    record_id: str,
    samples_before: int = 120,
    samples_after: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses and segments beats from an individual MIT-BIH record.

    Args:
        raw_dir: Directory containing raw files (.dat, .hea, .atr).
        record_id: MIT-BIH record number (e.g. '100').
        samples_before: Samples prior to R-peak (120 at 360 Hz ≈ 333 ms).
        samples_after: Samples following R-peak (240 at 360 Hz ≈ 667 ms).

    Returns:
        X: Extracted windows of shape (n_beats, 360).
        y: AAMI class integer labels of shape (n_beats,).
    """
    record_path = raw_dir / record_id
    signal, fs = load_raw_lead_ii(record_path)

    # 1. Bandpass filter (0.5 - 45 Hz)
    filtered = apply_bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=fs, order=3)

    # 2. Z-score normalization
    normalized = apply_zscore_norm(filtered)

    # 3. Read expert annotations (.atr)
    annotation = wfdb.rdann(str(record_path), "atr")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    window_length = samples_before + samples_after  # 360
    signal_length = len(normalized)

    for r_peak, symbol in zip(annotation.sample, annotation.symbol, strict=False):
        if symbol not in AAMI_MAPPING:
            continue

        # Boundary check: discard incomplete windows at start/end of recording
        if r_peak - samples_before < 0 or r_peak + samples_after > signal_length:
            continue

        beat_window = normalized[r_peak - samples_before : r_peak + samples_after]
        if len(beat_window) != window_length:
            continue

        aami_class = AAMI_MAPPING[symbol]
        class_idx = AAMI_CLASS_TO_INT[aami_class]

        windows.append(beat_window)
        labels.append(class_idx)

    if not windows:
        return np.empty((0, window_length), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.stack(windows, axis=0).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def process_split(
    record_ids: list[str],
    raw_dir: Path,
    split_name: str,
    samples_before: int = 120,
    samples_after: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Processes all records in a split and concatenates tensors."""
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    logger.info("Processing %s (%d records)...", split_name, len(record_ids))

    for record_id in record_ids:
        if record_id in EXCLUDED_PACEMAKER_RECORDS:
            logger.warning("Skipping excluded pacemaker record %s.", record_id)
            continue

        X_rec, y_rec = segment_record_beats(
            raw_dir=raw_dir,
            record_id=record_id,
            samples_before=samples_before,
            samples_after=samples_after,
        )

        if len(X_rec) > 0:
            all_windows.append(X_rec)
            all_labels.append(y_rec)
            logger.info("  Record %s: extracted %d beats.", record_id, len(X_rec))
        else:
            logger.warning("  Record %s: no beats extracted.", record_id)

    if not all_windows:
        total_len = samples_before + samples_after
        return np.empty((0, total_len), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_all = np.concatenate(all_windows, axis=0)
    y_all = np.concatenate(all_labels, axis=0)

    # Compute and log class breakdown
    counts = {name: int(np.sum(y_all == idx)) for name, idx in AAMI_CLASS_TO_INT.items()}
    logger.info("%s total beats: %d | Class distribution: %s", split_name, len(y_all), counts)

    return X_all, y_all


def main() -> None:
    """Entrypoint for data preprocessing stage."""
    # Enforce deterministic behavior
    np.random.seed(42)

    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting MIT-BIH preprocessing stage.")
    logger.info("Raw input directory: %s", raw_dir)
    logger.info("Processed output directory: %s", processed_dir)

    # Process DS1 (Training / Validation)
    X_train_ds1, y_train_ds1 = process_split(
        record_ids=DE_CHAZAL_DS1,
        raw_dir=raw_dir,
        split_name="DS1 (Train/Val)",
    )

    # Process DS2 (Independent Test)
    X_test_ds2, y_test_ds2 = process_split(
        record_ids=DE_CHAZAL_DS2,
        raw_dir=raw_dir,
        split_name="DS2 (Independent Test)",
    )

    # Save arrays in .npy format
    out_x_train = processed_dir / "X_train_ds1.npy"
    out_y_train = processed_dir / "y_train_ds1.npy"
    out_x_test = processed_dir / "X_test_ds2.npy"
    out_y_test = processed_dir / "y_test_ds2.npy"

    np.save(out_x_train, X_train_ds1)
    np.save(out_y_train, y_train_ds1)
    np.save(out_x_test, X_test_ds2)
    np.save(out_y_test, y_test_ds2)

    logger.info("Saved DS1 train features to %s (shape: %s)", out_x_train, X_train_ds1.shape)
    logger.info("Saved DS1 train labels to %s (shape: %s)", out_y_train, y_train_ds1.shape)
    logger.info("Saved DS2 test features to %s (shape: %s)", out_x_test, X_test_ds2.shape)
    logger.info("Saved DS2 test labels to %s (shape: %s)", out_y_test, y_test_ds2.shape)
    logger.info("Preprocessing stage completed successfully.")


if __name__ == "__main__":
    main()
