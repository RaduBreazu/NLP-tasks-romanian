from abc import ABC, abstractmethod

from typing import Dict

import pandas as pd

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from .plain import PlainNetwork
from .contrastive_learning import ContrastiveNetwork

class SelfTrainingModel(ABC, pl.LightningModule):
    def __init__(self, model: pl.LightningModule, labelled_data: Dataset, unlabelled_data: Dataset, inner_epochs: int, logging: bool) -> None:
        super().__init__()
        self.model = model
        self.labelled_data = labelled_data
        self.unlabelled_data = unlabelled_data
        self.inner_epochs = inner_epochs
        
        self.automatic_optimization = isinstance(model, PlainNetwork) or isinstance(model, ContrastiveNetwork)
        self.batch_size = self.model.batch_size
        self.num_labels = self.model.num_labels
        self.logging = logging

    def training_dataloader(self) -> DataLoader:
        if self.trainer is not None and self.current_epoch > 0 and self.current_epoch % self.inner_epochs == 0:
            self.model.eval()
            curr_loader = None
            if self.trainer is not None and hasattr(self.trainer, "train_dataloader"):
                curr_loader = self.trainer.train_dataloader

            if curr_loader is not None:
                with torch.no_grad():
                    new_data = self.labelled_data.data
                    unlabelled_dataloader = DataLoader(self.unlabelled_data, batch_size = curr_loader.batch_size, shuffle = False, collate_fn = self.unlabelled_data.collator)
                    pseudo_labelled_examples = []
                    all_high_conf_indices = []
                    
                    for batch_idx, batch in enumerate(unlabelled_dataloader):
                        out = self.model(batch) # the `forward` method of the model must return preds, probs and loss (if available) in this order
                        preds = out[0]
                        probs = out[1]
                        
                        # if the probability for a certain example is above a certain threshold (say 0.95), label the example with the predicted tag
                        prob_mask = probs >= 0.95
                        if prob_mask.any():
                            high_conf_indices = torch.where(prob_mask)[0].cpu().numpy()
                            for idx in high_conf_indices:
                                all_high_conf_indices.append(batch_idx * self.batch_size + idx)
                                example = {k : v[idx] for k, v in batch.items() if k != "label"}
                                example["label"] = preds[idx].item()
                                pseudo_labelled_examples.append(example)

                new_data = pd.concat([new_data, pd.DataFrame(pseudo_labelled_examples)], axis = 0, ignore_index = True)
                
                # update the labelled data subset by adding the newly added tags
                loader_args = {
                    "batch_size": curr_loader.batch_size,
                    "num_workers": curr_loader.num_workers,
                    "persistent_workers": curr_loader.persistent_workers,
                    "pin_memory": curr_loader.pin_memory,
                    "shuffle": curr_loader.shuffle,
                    "collate_fn": curr_loader.collator,
                    "generator": curr_loader.generator
                }
                self.labelled_data.data = new_data
                self.unlabelled_data.data = self.unlabelled_data.data.drop(index = all_high_conf_indices).reset_index(drop = True)
                return DataLoader(self.labelled_data, **loader_args)
            else:
                return DataLoader(self.labelled_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.labelled_data.collator)
        
        else:
            if self.trainer is not None:
                if hasattr(self.trainer, "train_dataloader"):
                    return self.trainer.train_dataloader
                
                return DataLoader(self.labelled_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.labelled_data.collator)
            
            else:
                return DataLoader(self.labelled_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.labelled_data.collator)

    def perform_step(self, batch, batch_idx: int, step_type: str) -> Dict[str, torch.Tensor]:
        match step_type:
            case "Training":
                step_method = self.model.training_step
            case "Validation":
                step_method = self.model.validation_step
            case "Test":
                step_method = self.model.test_step
        
        output = step_method(batch, batch_idx)
        if self.logging:
            y_pred = output["pred"]
            y_true = output["true"]
            self.compute_and_log_metrics(y_pred, y_true, step_type)

        return output
    
    def training_step(self, batch, batch_idx: int):
        return self.perform_step(batch, batch_idx, "Training")

    def validation_step(self, batch, batch_idx: int):
        return self.perform_step(batch, batch_idx, "Validation")
    
    def test_step(self, batch, batch_idx: int):
        return self.perform_step(batch, batch_idx, "Test")
    
    @abstractmethod
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        pass
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()