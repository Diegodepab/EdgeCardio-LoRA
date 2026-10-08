"""PyTorch Dataset and DataLoader utilities for ECG beat classification."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ECGDataset(Dataset):
    """PyTorch Dataset for single-lead ECG beat windows.

    Yields:
        signal: 1D signal tensor with shape (1, 360), dtype float32.
        label: Class label tensor with shape (), dtype long (0 to 4).
    """

    def __init__(
        self,
        signals: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> None:
        """Initializes ECGDataset.

        Args:
            signals: Array of shape (N, 360) or (N, 1, 360).
            labels: Array of shape (N,) containing class indices.
        """
        if len(signals) != len(labels):
            raise ValueError(
                f"Signals length ({len(signals)}) does not match labels length ({len(labels)})"
            )

        if isinstance(signals, np.ndarray):
            self.signals = torch.from_numpy(signals).float()
        else:
            self.signals = signals.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        # Guarantee shape (1, 360) for 1D convolution
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        label = self.labels[idx]
        return signal, label


def compute_class_weights(
    labels: np.ndarray | torch.Tensor,
    num_classes: int = 5,
) -> torch.Tensor:
    """Computes balanced inverse-frequency class weights for CrossEntropyLoss.

    Formula: w_c = N_total / (num_classes * N_c)
    Normalized so the mean of active classes equals 1.0.

    Args:
        labels: 1D array of class indices.
        num_classes: Total number of target classes (default: 5 for AAMI EC57).

    Returns:
        torch.FloatTensor of shape (num_classes,) with balanced weights.
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    total_samples = len(labels_np)
    counts = np.bincount(labels_np, minlength=num_classes)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total_samples / (num_classes * counts[c])
        else:
            weights[c] = 0.0

    # Normalize weights so mean of active weights is 1.0 (stabilizes loss scale)
    active = weights[weights > 0]
    if len(active) > 0:
        weights = weights / np.mean(active)

    return torch.from_numpy(weights).float()


def get_dataloaders(
    data_dir: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Loads processed .npy files and constructs train and validation DataLoaders.

    Args:
        data_dir: Path to directory containing processed numpy arrays.
                  Defaults to 'data/processed' relative to repository root.
        batch_size: Batch size for DataLoader.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning.

    Returns:
        Tuple of (train_loader, val_loader, class_weights).
    """
    if data_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_path = repo_root / "data" / "processed"
    else:
        data_path = Path(data_dir)

    train_x_path = data_path / "X_train_ds1.npy"
    train_y_path = data_path / "y_train_ds1.npy"
    val_x_path = data_path / "X_test_ds2.npy"
    val_y_path = data_path / "y_test_ds2.npy"

    for p in (train_x_path, train_y_path, val_x_path, val_y_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing required dataset file: {p}. Run preprocessing pipeline first."
            )

    x_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    x_val = np.load(val_x_path)
    y_val = np.load(val_y_path)

    train_dataset = ECGDataset(x_train, y_train)
    val_dataset = ECGDataset(x_val, y_val)

    class_weights = compute_class_weights(y_train, num_classes=5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, class_weights
