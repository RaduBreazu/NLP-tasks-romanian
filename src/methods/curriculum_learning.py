import warnings

from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

# The curriculum sampler
class CurriculumSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, difficulty_scores: List[float]) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        assert len(self.dataset) == len(self.difficulty_scores), (
            f"dataset len {len(self.dataset)} != difficulty len {len(self.difficulty_scores)}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        sorted_indices = sorted(
            range(len(self.difficulty_scores)),
            key = lambda i: self.difficulty_scores[i]
        )
        return iter(sorted_indices)


# The teacher model
class DifficultyPredictor(ABC, pl.LightningModule):
    def __init__(self, model_name : str):
        super(pl.LightningModule, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.config.hidden_size, 1)
        )
        self.criterion = F.mse_loss

    @abstractmethod
    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]):
        x = self.tokenize_input(batch)
        match x:
            case (input_ids, attention_mask, target_losses):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                target_losses = target_losses.to(self.device).float()

                out = self.model(input_ids, attention_mask = attention_mask).last_hidden_state
                out = self.linear(out[:, 0, :])
                out = out.squeeze(-1)
                loss = self.criterion(out, target_losses.view_as(out))
                return loss, out

            case (input_ids, attention_mask):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                out = self.model(input_ids, attention_mask = attention_mask).last_hidden_state
                out = self.linear(out[:, 0, :]).squeeze(-1)
                return out
            

# The student model
class CurriculumTagger(ABC, pl.LightningModule):
    def __init__(self, model_name : str, train_dataset: Dataset, num_labels: int, batch_size: int) -> None:
        super(pl.LightningModule, self).__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_labels = num_labels

        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.transformer.config.hidden_size, self.num_labels)
        )

    @abstractmethod
    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass

    # This method must also return the normalized logits (i.e. the classification probabilities)
    # Be aware that this method receives an already tokenized input
    @abstractmethod
    def forward(self, input_ids, attention_mask, labels) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass


# LightningModule that runs both the student and the teacher
class CurriculumLearningModel(ABC, pl.LightningModule):
    def __init__(self, student_model: pl.LightningModule, teacher_model: DifficultyPredictor, train_dataloader_args: Dict[str, Any], learning_rate: float,
                 logging : bool) -> None:
        super(pl.LightningModule, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.automatic_optimization = False
        self.train_dataloader_args = train_dataloader_args
        self.learning_rate = learning_rate

        self.logging = logging

    # Before each training epoch, we need to reorder the texts, according to the loss function of the teacher
    def train_dataloader(self):
        self.teacher_model.eval()
        all_scores = []
        with torch.no_grad():
            default_loader = DataLoader(
                self.student_model.train_dataset,
                batch_size = self.student_model.batch_size,
                collate_fn = type(self.student_model.train_dataset).collator,
            )
            for batch in default_loader:
                # Obtain difficulty scores for the dataset from the teacher model
                scores = self.teacher_model(batch)
                all_scores.append(scores.cpu())

        difficulty_scores = torch.cat(all_scores).tolist()
        sampler = CurriculumSampler(self.student_model.train_dataset, difficulty_scores) # this is where the examples get reordered

        return DataLoader(self.student_model.train_dataset, sampler = sampler, **self.train_dataloader_args)

    def forward(self, batch : Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_ids, attention_mask, labels = self.student_model.tokenize_input(batch)
        return self.student_model(input_ids, attention_mask, labels)
    
    # Auxiliary function that computes and logs the performance metrics
    @abstractmethod
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        pass

    # This method computes the total loss of the model from the returned loss (useful for when embedding contrastive learning)
    @abstractmethod
    def compute_total_loss(self, losses) -> torch.Tensor:
        pass
    
    # This method computes per-sample loss for teacher training
    @abstractmethod
    def compute_per_sample_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        student_opt, teacher_opt = self.optimizers()

        input_ids, attention_mask, labels = self.student_model.tokenize_input(batch)
        batch_size = input_ids.size(0)

        # 1) Train student
        self.student_model.train()
        student_opt.zero_grad()

        student_preds, student_probs, student_losses = self.student_model(input_ids, attention_mask, labels)
        student_loss = self.compute_total_loss(student_losses)

        self.manual_backward(student_loss)
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm = 1.0)
        student_opt.step()

        # 2) Train teacher - compute per-sample loss
        self.teacher_model.train()
        teacher_opt.zero_grad()

        target_losses = self.compute_per_sample_loss(input_ids, attention_mask, labels)
        teacher_input_ids, teacher_attention_mask, _ = self.teacher_model.tokenize_input(batch)
        teacher_loss, teacher_output = self.teacher_model((teacher_input_ids, teacher_attention_mask, target_losses))

        self.manual_backward(teacher_loss)
        torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), max_norm = 1.0)
        teacher_opt.step()

        # 3) Logging
        if self.logging:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.log("Training loss", student_loss.detach().item(), prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
                self.log("Teacher loss", teacher_loss.detach().item(), prog_bar = False, batch_size = batch_size, on_step = False, on_epoch = True)

            # Metrics
            self.compute_and_log_metrics(student_preds, labels, "Training")

        return student_loss

    
    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self.student_model.tokenize_input(batch)
        batch_size = input_ids.size(0)

        student_loss, student_output = self.student_model(input_ids, attention_mask, labels)

        if self.logging:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.log("Test loss", student_loss.detach().item(), prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)

            self.compute_and_log_metrics(student_output, labels, "Test")
        
        return student_loss

    # Optimizers + schedulers
    def configure_optimizers(self):
        student_opt = optim.AdamW(self.student_model.parameters(), lr = self.learning_rate, weight_decay = 0.01)
        teacher_opt = optim.AdamW(self.teacher_model.parameters(), lr = self.learning_rate, weight_decay = 0.01)

        student_lr_scheduler = optim.lr_scheduler.StepLR(student_opt, step_size = 1, gamma = 0.95)
        teacher_lr_scheduler = optim.lr_scheduler.StepLR(teacher_opt, step_size = 1, gamma = 0.95)

        return (
            [student_opt, teacher_opt],
            [
                {"scheduler": student_lr_scheduler, "interval": "epoch", "name": "student_lr"},
                {"scheduler": teacher_lr_scheduler, "interval": "epoch", "name": "teacher_lr"}
            ]
        )