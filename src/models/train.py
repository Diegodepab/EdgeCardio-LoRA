"""Training pipeline for ECG baseline model with MLflow experiment tracking.

Implements:
1. Dynamic class weighting for CrossEntropyLoss to address severe AAMI class imbalance.
2. Tracking of training loss, validation loss, and Macro F1-Score per epoch.
3. Checkpoint persistence of the best model state dict based on Macro F1.
4. Experiment logging in MLflow with a local SQLite backend.
"""

import argparse
import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import get_dataloaders
from src.models.baseline_cnn import ECG1DCNN, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baseline")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic behavior across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> tuple[float, dict[str, float]]:
    """Calculates macro-averaged F1-Score and per-class F1 scores.

    Args:
        y_true: 1D array of ground truth labels.
        y_pred: 1D array of predicted class labels.
        num_classes: Total number of classes (default: 5).

    Returns:
        macro_f1: Macro-averaged F1-Score.
        per_class_f1: Dictionary mapping class name to F1 score.
    """
    per_class_f1: dict[str, float] = {}
    f1_values: list[float] = []

    for c in range(num_classes):
        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        total_true = int(np.sum(y_true == c))
        if total_true == 0:
            per_class_f1[class_name] = 0.0
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        per_class_f1[class_name] = f1
        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return macro_f1, per_class_f1


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs a single training epoch and returns mean loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)
        total_samples += len(labels)

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 5,
) -> tuple[float, float, dict[str, float]]:
    """Evaluates model on validation/test dataset.

    Returns:
        val_loss: Average CrossEntropyLoss on validation set.
        macro_f1: Macro-averaged F1-Score across all classes.
        per_class_f1: Dictionary with F1 score for each AAMI class.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * len(labels)
            total_samples += len(labels)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64)

    macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=num_classes)
    return val_loss, macro_f1, per_class_f1


def train_baseline(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
) -> float:
    """Executes the full baseline training and validation pipeline with MLflow tracking.

    Returns:
        best_val_macro_f1: Highest Macro F1-Score achieved on validation set.
    """
    set_seed(seed)
    repo_root = Path(__file__).resolve().parents[2]

    if data_dir is None:
        data_dir = repo_root / "data" / "processed"
    if output_dir is None:
        output_dir = repo_root / "models" / "baseline"

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "baseline.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Execution device: %s", device)

    # 1. Load data and compute class weights
    train_loader, val_loader, class_weights = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    class_weights = class_weights.to(device)
    logger.info(
        "Computed balanced class weights: %s", class_weights.cpu().numpy().round(4).tolist()
    )

    # 2. Instantiate model, loss, and optimizer
    model = ECG1DCNN(in_channels=1, num_classes=5, dropout_rate=0.2).to(device)
    param_info = count_parameters(model)
    logger.info("Initialized ECG1DCNN with %d total parameters.", param_info["total_params"])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Configure local MLflow tracking
    tracking_uri = f"sqlite:///{repo_root / 'mlruns.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ECG_Baseline")
    logger.info("MLflow tracking initialized at URI: %s", tracking_uri)

    best_macro_f1 = 0.0
    best_epoch = -1

    with mlflow.start_run(run_name="baseline_1d_cnn_inter_patient"):
        # Log training hyperparameters
        mlflow.log_params(
            {
                "model_architecture": "ECG1DCNN",
                "total_parameters": param_info["total_params"],
                "trainable_parameters": param_info["trainable_params"],
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": "AdamW",
                "loss_function": "WeightedCrossEntropyLoss",
                "seed": seed,
                "class_weights": class_weights.cpu().numpy().round(4).tolist(),
            }
        )

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_macro_f1, per_class_f1 = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=5,
            )

            metrics_to_log = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
            }
            for cls_name, f1_val in per_class_f1.items():
                metrics_to_log[f"val_f1_{cls_name}"] = f1_val

            mlflow.log_metrics(metrics_to_log, step=epoch)

            f1_str = " | ".join(f"{k}: {v:.3f}" for k, v in per_class_f1.items())
            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f (%s)",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_macro_f1,
                f1_str,
            )

            # Checkpoint model if validation Macro F1 improves
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metric("best_val_macro_f1", best_macro_f1, step=epoch)
                logger.info(
                    "--> New best model checkpoint saved to %s (Macro F1 = %.4f at epoch %d)",
                    best_model_path,
                    best_macro_f1,
                    best_epoch,
                )

        logger.info(
            "Training completed. Best Val Macro F1: %.4f (Epoch %d). Checkpoint: %s",
            best_macro_f1,
            best_epoch,
            best_model_path,
        )

        if best_model_path.is_file():
            mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")

    return best_macro_f1


def main() -> None:
    """CLI entrypoint for baseline model training."""
    parser = argparse.ArgumentParser(description="Train ECG 1D-CNN Baseline with MLflow tracking")
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs (default: 25)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing processed .npy files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save baseline.pt checkpoint"
    )
    args = parser.parse_args()

    train_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
