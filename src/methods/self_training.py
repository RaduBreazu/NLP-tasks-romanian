import pandas as pd

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class SelfTrainingModel(pl.LightningModule):
    def __init__(self, model: pl.LightningModule, labelled_data: Dataset, unlabelled_data: Dataset) -> None:
        super(pl.LightningModule, self).__init__()
        self.model = model
        self.labelled_data = labelled_data
        self.unlabelled_data = unlabelled_data

    def training_dataloader(self) -> DataLoader:
        self.model.eval()
        curr_loader = None
        if self.trainer is not None and hasattr(self.trainer, "train_dataloader"):
            curr_loader = self.trainer.train_dataloader

        if curr_loader is not None:
            with torch.no_grad():
                new_data = self.labelled_data.data
                unlabelled_dataloader = DataLoader(self.unlabelled_data, batch_size = curr_loader.batch_size, collate_fn = self.unlabelled_data.collator)
                pseudo_labelled_examples = []
                
                for batch in unlabelled_dataloader:
                    out = self.model(batch) # the `forward` method of the model must return preds, probs and loss (if available) in this order
                    preds = out[0]
                    probs = out[1]
                    
                    # if the probability for a certain example is above a certain threshold (say 0.8), label the example with the predicted tag
                    prob_mask = probs >= 0.8
                    if prob_mask.any():
                        high_conf_indices = torch.where(prob_mask)[0].cpu().numpy()
                        for idx in high_conf_indices:
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
            self.unlabelled_data.data = self.unlabelled_data.data.drop(index = high_conf_indices)
            new_loader = DataLoader(self.labelled_data, **loader_args)
        else:
            new_loader = DataLoader(self.labelled_data, batch_size = 32, shuffle = True, collate_fn = self.labelled_data.collator)

        return new_loader

    def training_step(self, batch, batch_idx: int):
        return self.model.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        return self.model.test_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()