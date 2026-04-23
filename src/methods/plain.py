from typing import Tuple, Dict

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer

class PlainNetwork(ABC, pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, batch_size: int, learning_rate: float, logging : bool) -> None:
        super(pl.LightningModule, self).__init__()
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
    def tokenize_input(self, batch) -> Tuple[torch.Tensor. torch.Tensor, torch.Tensor]:
        pass
    
    # This method must also return the normalized logits (i.e. the classification probabilities)
    @abstractmethod
    def forward(self, input_ids, attention_mask, labels):
        pass
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self.tokenize_input(batch)
        preds, probs, loss = self(input_ids, attention_mask, labels)

        if self.logging:
            self.log("Train loss", loss, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)
            self.compute_and_log_metrics(preds, labels, "Training")

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self.tokenize_input(labels)
        preds, probs, loss = self(input_ids, attention_mask, labels)

        if self.logging:
            self.log("Test loss", loss, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)
            self.compute_and_log_metrics(preds, labels, "Test")

        return loss
    
    @abstractmethod
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay = 0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
        return [optimizer], [scheduler]