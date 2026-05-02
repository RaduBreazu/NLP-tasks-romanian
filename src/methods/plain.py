from typing import Tuple, Dict, Optional

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer

class PlainNetwork(ABC, pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, batch_size: int, learning_rate: float, logging : bool) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels

        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.transformer.config.hidden_size, self.num_labels)
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index = -100)
        self.logging = logging

    @abstractmethod
    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass
    
    @abstractmethod
    def forward(self, batch : Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass

    # This method is used when embedding a plain model in another, more complex one
    def compute_total_loss(self, loss : Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return loss

    def perform_step(self, batch : Dict[str, torch.Tensor], step_type : str) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = self.tokenize_input(batch)
        new_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
        preds, probs, loss = self(new_batch)

        if self.logging and loss is not None:
            self.log(f"{step_type} loss", loss, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)
            self.compute_and_log_metrics(preds, labels, "Training" if step_type == "Train" else step_type)
        
        return {"pred": preds, "true": labels, "loss": loss}
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.perform_step(batch, "Training")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.perform_step(batch, "Validation")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.perform_step(batch, "Test")
    
    @abstractmethod
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay = 0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
        return [optimizer], [scheduler]