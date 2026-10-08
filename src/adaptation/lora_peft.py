"""LoRA (Low-Rank Adaptation) module for ECG personal patient calibration.

Wraps the pre-trained ECG1DCNN baseline model with HuggingFace PEFT,
freezing all base model weights (requires_grad = False) and injecting
trainable low-rank decomposition matrices (A and B) into the classifier.
"""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from src.models.baseline_cnn import ECG1DCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lora_peft")


def load_baseline_model(
    weights_path: Path | str | None = None,
    device: torch.device | None = None,
) -> ECG1DCNN:
    """Instantiates ECG1DCNN and loads pre-trained baseline checkpoint if available.

    Args:
        weights_path: Path to baseline.pt state dictionary.
        device: Target execution device.

    Returns:
        ECG1DCNN instance.
    """
    model = ECG1DCNN(in_channels=1, num_classes=5, dropout_rate=0.2)

    if weights_path is not None:
        p = Path(weights_path)
        if p.is_file():
            state_dict = torch.load(p, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded pre-trained baseline weights from %s", p)
        else:
            logger.warning(
                "Baseline checkpoint %s not found. Proceeding with uninitialized weights.", p
            )

    if device is not None:
        model.to(device)

    return model


def create_peft_model(
    base_model: ECG1DCNN,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> PeftModel:
    """Wraps ECG1DCNN with HuggingFace PEFT LoRA adapters.

    All base model layers are completely frozen (requires_grad = False).
    Trainable parameters are strictly limited to the injected low-rank matrices.

    Args:
        base_model: Pre-trained ECG1DCNN model.
        r: LoRA decomposition rank (default: 4).
        lora_alpha: LoRA scaling factor (default: 8).
        lora_dropout: Dropout probability for LoRA layers (default: 0.05).
        target_modules: List of module names where LoRA is injected.
                        Defaults to ['classifier.0'] (the 128 -> 64 latent projection layer).

    Returns:
        PeftModel wrapping the base network.
    """
    if target_modules is None:
        # Injects LoRA into the latent feature projection layer of the classifier
        target_modules = ["classifier.0"]

    # 1. Guarantee that all base model weights are 100% frozen
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Define PEFT LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # 3. Inject adapters
    peft_model = get_peft_model(base_model, lora_config)

    logger.info(
        "Injected LoRA adapters (r=%d, alpha=%d, target_modules=%s).", r, lora_alpha, target_modules
    )
    peft_model.print_trainable_parameters()

    return peft_model
