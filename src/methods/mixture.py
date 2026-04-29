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
    def __init__(self, models: List[pl.LightningModule], batch_size : int, num_labels : int, voting_mechanism : VotingMechanism, learning_rate: float, logging : bool) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.tokenizers = [model.tokenizer for model in models]
        self.models = torch.nn.ModuleList(models)

        self.automatic_optimization = False
        self.voting_mechanism = voting_mechanism
        self.learning_rate = learning_rate
        self.logging = logging

    # Since it is called by the `forward` method of the concrete model, this method must return a tuple of the form `(preds, probs, loss)`
    # Be aware of the fact that the loss computed here must be differentiable!
    def forward(self, x):
        all_preds = []
        all_probs = []
        all_losses = []
        for model in self.models:
            input_ids, attention_mask, labels = model.tokenize_input(x)
            x_prime = {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
            preds, probs, loss = model(x_prime)
            all_preds.append(preds)
            all_probs.append(probs)
            all_losses.append(loss)

        all_probs = torch.stack(all_probs)
        all_losses = torch.stack(all_losses)
        return self.voting_mechanism(all_probs, all_losses)
    
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

            input_ids, attention_mask, labels = model.tokenize_input(batch)
            new_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
            preds, probs, loss = model(new_batch)
            all_preds.append(preds)
            all_probs.append(probs)
            all_losses.append(loss)
        
        all_probs = torch.stack(all_probs)
        all_losses = torch.stack(all_losses)
        preds, probs, loss = self.voting_mechanism(all_probs, all_losses)
        
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
            self.compute_and_log_metrics(preds, labels, step_type)
    
    def validation_step(self, batch, batch_idx : int) -> None:
        return self.evaluation_step(batch, "Validation")

    def test_step(self, batch, batch_idx : int) -> None:
        return self.evaluation_step(batch, "Test")

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        for model in self.models:
            opt = torch.optim.AdamW(model.parameters(), lr = self.learning_rate, weight_decay = 0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size = 1, gamma = 0.95)
            optimizers.append(opt)
            schedulers.append({"scheduler": sched, "interval": "epoch", "frequency": 1})

        return optimizers, schedulers