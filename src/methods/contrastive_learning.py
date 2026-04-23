from typing import Tuple, Dict

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, classifier_weights : torch.Tensor) -> torch.Tensor:
        """
        Full Dual Contrastive Loss
        Args:
            embeddings: [N, D] normalized embeddings for the batch
            labels: [N] class labels for the batch
            classifier_weights: [C, D] weights of the linear layer
        """
        # 1. Normalize
        features = F.normalize(embeddings, p = 2, dim = 1)
        weights = F.normalize(classifier_weights, p = 2, dim = 1)
        
        # 2. Compute similarity matrix [N, C]
        sim_matrix = torch.matmul(features, weights.T) / self.temperature
        
        # --- L_z: Sample-to-Label Loss ---
        # For each sample (row), we want to identify the correct class (column)
        # This is equivalent to CrossEntropy across the rows
        loss_z = F.cross_entropy(sim_matrix, labels)
        
        # --- L_theta: Label-to-Sample Loss ---
        # For each class (column), we want to identify which samples (rows) belong to it
        # We transpose the matrix to get [C, N] and use a mask for correct samples
        sim_matrix_theta = sim_matrix.T # [C, N]
        
        # Create a binary mask where mask[j, i] = 1 if sample i has label j
        # Shape: [C, N]
        adj_mask = (labels.unsqueeze(0) == torch.arange(weights.size(0), device = labels.device).unsqueeze(1)).float()
        
        # We use a log-sum-exp approach for the denominator across samples
        log_prob_theta = F.log_softmax(sim_matrix_theta, dim = 1)
        
        # Multiply by mask to keep only correct sample-label pairs, then average
        # We divide by the sum of the mask to get the mean per class that exists in the batch
        loss_theta = -(adj_mask * log_prob_theta).sum(dim = 1) / (adj_mask.sum(dim = 1) + 1e-10)
        loss_theta = loss_theta.mean()

        # The total dual loss is the sum of the previous losses (L_z and L_theta)
        return loss_z + loss_theta
    

class ContrastiveNetwork(ABC, pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, batch_size: int, learning_rate: float, contrastive_weight: float, temperature: float, logging : bool) -> None:
        super(pl.LightningModule, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contrastive_weight = contrastive_weight
        self.num_labels = num_labels

        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.transformer.config.hidden_size, self.num_labels)
        )

        # Projection head for contrastive learning
        hidden_size = self.transformer.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = -100)
        self.contrastive_loss = ContrastiveLoss(temperature)

        self.logging = logging

    @abstractmethod
    def tokenize_input(self, batch) -> Tuple[torch.Tensor. torch.Tensor, torch.Tensor]:
        pass
    
    # This method must also return the normalized logits (i.e. the classification probabilities)
    @abstractmethod
    def forward(self, input_ids, attention_mask, labels, compute_contrastive = True):
        pass
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self.tokenize_input(batch)
        preds, probs, (ce_loss, cont_loss) = self(input_ids, attention_mask, labels, compute_contrastive = True)
        loss = ce_loss + self.contrastive_weight * cont_loss

        if self.logging:
            self.log("Train loss", loss, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)
            self.log("Train cross-entropy loss", ce_loss, prog_bar = False, on_step = False, on_epoch = True, batch_size = self.batch_size)
            if isinstance(cont_loss, torch.Tensor):
                self.log("Train contrastive loss", cont_loss, prog_bar = False, on_step = False, on_epoch = True, batch_size = self.batch_size)

            self.compute_and_log_metrics(preds, labels, "Training")

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self.tokenize_input(labels)
        preds, probs, (ce_loss, _) = self(input_ids, attention_mask, labels, compute_contrastive = False)
        loss = ce_loss

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