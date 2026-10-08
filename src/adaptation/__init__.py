"""Parameter-Efficient Fine-Tuning (LoRA) module for patient-specific ECG adaptation."""

from src.adaptation.lora_peft import create_peft_model

__all__ = ["create_peft_model"]
