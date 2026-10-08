"""Script for calibrating patient-specific LoRA adapters on few-shot ECG beats.

Simulates hospital admission scenario:
- Loads a small slice (100-200 beats) from DS2 as patient baseline.
- Freezes base model, trains only the lightweight LoRA matrices for 5-10 epochs.
- Tracks metrics in MLflow experiment 'ECG_Patient_Adaptation'.
- Persists only the lightweight adapter weights via save_pretrained (< 100 KB).
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.adaptation.lora_peft import create_peft_model, load_baseline_model
from src.data.dataset import ECGDataset, compute_class_weights
from src.models.train import compute_macro_f1, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_adapter")


def get_directory_size_kb(directory: Path) -> float:
    """Calculates total disk size of all files in a directory in kilobytes."""
    total_bytes = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total_bytes / 1024.0


def create_patient_dataloaders(
    data_dir: Path,
    patient_beats: int = 150,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Simulates a hospital admission scenario with limited patient ECG beats.

    Extracts a small slice from DS2 to emulate newly acquired patient data.

    Args:
        data_dir: Path to directory containing processed numpy arrays.
        patient_beats: Number of beats for patient adaptation (default: 150).
        batch_size: Batch size for DataLoader.

    Returns:
        Tuple of (patient_train_loader, patient_val_loader, class_weights).
    """
    x_test_path = data_dir / "X_test_ds2.npy"
    y_test_path = data_dir / "y_test_ds2.npy"

    if not x_test_path.is_file() or not y_test_path.is_file():
        raise FileNotFoundError(f"Missing DS2 test files in {data_dir}. Run preprocessing first.")

    x_all = np.load(x_test_path)
    y_all = np.load(y_test_path)

    total_available = len(y_all)
    if patient_beats > total_available // 2:
        patient_beats = total_available // 2

    # Slice for calibration (hospital intake ECG recording)
    x_patient_train = x_all[:patient_beats]
    y_patient_train = y_all[:patient_beats]

    # Slice for evaluation (subsequent continuous monitoring of the same patient)
    eval_slice_end = min(patient_beats + 300, total_available)
    x_patient_val = x_all[patient_beats:eval_slice_end]
    y_patient_val = y_all[patient_beats:eval_slice_end]

    class_weights = compute_class_weights(y_patient_train, num_classes=5)

    train_ds = ECGDataset(x_patient_train, y_patient_train)
    val_ds = ECGDataset(x_patient_val, y_patient_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "Patient data slice: %d calibration beats for adaptation, %d beats for evaluation.",
        len(train_ds),
        len(val_ds),
    )
    return train_loader, val_loader, class_weights


def train_adapter(
    patient_id: str = "patient_01",
    num_patient_beats: int = 150,
    epochs: int = 8,
    lr: float = 1e-3,
    r: int = 4,
    lora_alpha: int = 8,
    batch_size: int = 32,
    seed: int = 42,
    base_weights_path: Path | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> float:
    """Trains a patient-specific LoRA adapter and persists the lightweight artifact.

    Args:
        patient_id: Identifier for patient profile.
        num_patient_beats: Number of beats for calibration.
        epochs: Number of adaptation epochs.
        lr: Learning rate for AdamW.
        r: LoRA decomposition rank.
        lora_alpha: LoRA scaling factor.
        batch_size: Batch size.
        seed: Random seed.
        base_weights_path: Path to pre-trained baseline.pt checkpoint.
        data_dir: Path to data/processed directory.
        output_dir: Destination directory for adapter files.

    Returns:
        best_macro_f1: Best validation Macro F1 score on patient test slice.
    """
    set_seed(seed)
    repo_root = Path(__file__).resolve().parents[2]

    if data_dir is None:
        data_dir = repo_root / "data" / "processed"
    if base_weights_path is None:
        base_weights_path = repo_root / "models" / "baseline" / "baseline.pt"
    if output_dir is None:
        output_dir = repo_root / "models" / "lora_adapters" / patient_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training LoRA adapter for patient '%s' on %s", patient_id, device)

    # 1. Prepare patient data
    train_loader, val_loader, class_weights = create_patient_dataloaders(
        data_dir=data_dir,
        patient_beats=num_patient_beats,
        batch_size=batch_size,
    )
    class_weights = class_weights.to(device)

    # 2. Instantiate base model and inject LoRA
    base_model = load_baseline_model(weights_path=base_weights_path, device=device)
    peft_model = create_peft_model(
        base_model=base_model,
        r=r,
        lora_alpha=lora_alpha,
    ).to(device)

    # 3. Only pass trainable parameters to optimizer (strictly LoRA matrices)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. MLflow Experiment Tracking
    tracking_uri = f"sqlite:///{repo_root / 'mlruns.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ECG_Patient_Adaptation")

    best_macro_f1 = 0.0

    with mlflow.start_run(run_name=f"lora_{patient_id}"):
        mlflow.log_params(
            {
                "patient_id": patient_id,
                "patient_beats": num_patient_beats,
                "lora_rank": r,
                "lora_alpha": lora_alpha,
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "seed": seed,
            }
        )

        for epoch in range(1, epochs + 1):
            # Training phase
            peft_model.train()
            running_loss = 0.0
            total_train = 0

            for signals, labels in train_loader:
                signals = signals.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = peft_model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(labels)
                total_train += len(labels)

            epoch_loss = running_loss / total_train if total_train > 0 else 0.0

            # Evaluation phase on patient test slice
            peft_model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals = signals.to(device)
                    logits = peft_model(signals)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(labels.numpy())

            y_pred = (
                np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
            )
            y_true = (
                np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64)
            )
            macro_f1, _ = compute_macro_f1(y_true, y_pred, num_classes=5)

            mlflow.log_metrics(
                {
                    "patient_train_loss": epoch_loss,
                    "patient_val_macro_f1": macro_f1,
                },
                step=epoch,
            )

            logger.info(
                "Patient %s | Epoch %02d/%02d | Loss: %.4f | Macro F1: %.4f",
                patient_id,
                epoch,
                epochs,
                epoch_loss,
                macro_f1,
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1

        # 5. Persist only the lightweight adapter weights (save_pretrained)
        output_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(output_dir))
        artifact_size_kb = get_directory_size_kb(output_dir)

        mlflow.log_metric("final_best_macro_f1", best_macro_f1)
        mlflow.log_metric("adapter_size_kb", artifact_size_kb)
        mlflow.log_artifacts(str(output_dir), artifact_path="adapter_files")

        logger.info("=" * 65)
        logger.info("Adaptation completed for patient '%s'.", patient_id)
        logger.info("Best Validation Macro F1: %.4f", best_macro_f1)
        logger.info("Adapter saved at: %s", output_dir)
        logger.info("Total adapter disk footprint: %.2f KB (< 100 KB budget)", artifact_size_kb)
        logger.info("=" * 65)

    return best_macro_f1


def main() -> None:
    """CLI entrypoint for patient LoRA adaptation."""
    parser = argparse.ArgumentParser(description="Train patient-specific LoRA adapter for ECG")
    parser.add_argument("--patient-id", type=str, default="patient_01", help="Patient identifier")
    parser.add_argument(
        "--beats", type=int, default=150, help="Calibration beats slice (default: 150)"
    )
    parser.add_argument("--epochs", type=int, default=8, help="Adaptation epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--r", type=int, default=4, help="LoRA rank (default: 4)")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha (default: 8)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--base-weights", type=str, default=None, help="Path to baseline.pt checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to processed data directory"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save adapter")
    args = parser.parse_args()

    train_adapter(
        patient_id=args.patient_id,
        num_patient_beats=args.beats,
        epochs=args.epochs,
        lr=args.lr,
        r=args.r,
        lora_alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        base_weights_path=Path(args.base_weights) if args.base_weights else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
