from typing import List
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl

class VotingMechanism(ABC):
    def __init__(self) -> None:
        return None
    
    @abstractmethod
    def __call__(self, preds, probs, losses) -> None:
        pass

class MixtureModel(ABC, pl.LightningModule):
    def __init__(self, models: List[pl.LightningModule], batch_size : int, voting_mechanism : VotingMechanism, logging : bool) -> None:
        super(pl.LightningModule, self).__init__()
        self.batch_size = batch_size
        self.tokenizers = [model.tokenizer for model in models]
        self.models = models

        self.automatic_optimization = False
        self.voting_mechanism = voting_mechanism
        self.logging = logging

    # Since it is called by the `forward` method of the concrete model, this method must return a tuple of the form `(preds, probs, loss)`
    # Be aware of the fact that the loss computed here must be differentiable!
    def forward(self, x):
        all_preds = []
        all_probs = []
        all_losses = []
        for model in self.models:
            preds, probs, loss = model(x)
            all_preds.append(preds)
            all_probs.append(probs)
            all_losses.append(loss)

        return self.voting_mechanism(all_preds, all_probs, all_losses)
    
    # Auxiliary function that computes and logs the performance metrics
    @abstractmethod
    def compute_and_log_metrics(self, y_pred, y_true, step_type : str) -> None:
        pass

    def training_step(self, batch, batch_idx : int) -> None:
        optimizers = self.optimizers()
        
        all_preds = []
        all_probs = []
        all_losses = []
        for model, optimizer in zip(self.models, optimizers):
            model.train()
            optimizer.zero_grad()

            preds, probs, loss = model(batch)
            all_preds.append(preds)
            all_probs.append(probs)
            all_losses.append(loss)
        
        preds, probs, loss = self.voting_mechanism(all_preds, all_probs, all_losses)
        for model, optimizer in zip(self.models, optimizers):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
        
        labels = batch["label"]
        if self.logging:
            self.compute_and_log_metrics(preds, labels, "Training")

    def evaluation_step(self, batch, step_type : str) -> None:
        for model in self.models:
            model.eval()
        
        preds, probs, losses = self(batch)
        labels = batch["label"]
        if self.logging:
            self.compute_and_log_metrics(labels, preds, step_type)
    
    def validation_step(self, batch, batch_idx : int) -> None:
        return self.evaluation_step(self, batch, "Validation")

    def test_step(self, batch, batch_idx : int) -> None:
        return self.evaluation_step(self, batch, "Test")

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def configure_optimizers(self):
        optimizers = [model.optimizers() for model in self.models]
        schedulers = [model.lr_schedulers() for model in self.models]
        
        return optimizers, schedulers