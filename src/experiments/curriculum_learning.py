import argparse
import warnings
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

from functools import partial
from typing import Tuple, Dict, Optional, Callable, Any
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from seqeval.scheme import IOB2

from . import FakeNewsDataset, read_fake_news_dataset
from . import MarcellDataset, read_legal_ner_dataset

from . import DifficultyPredictor, CurriculumTagger, CurriculumLearningModel

from . import MyCallback

class NERTeacherModel(DifficultyPredictor):
    def __init__(self, model_name: str) -> None:
        super(DifficultyPredictor, self).__init__(model_name)

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "longest", truncation = True, return_offsets_mapping = True, is_split_into_words = True)
        texts = texts.to(self.device)
        try:
            labels = batch["label"]
            labels = pad_sequence(labels, batch_first = True, padding_value = 0).to(self.device)
        except KeyError:
            return texts["input_ids"], texts["attention_mask"]
        
        return texts["input_ids"], texts["attention_mask"], labels


class NERStudentModel(CurriculumTagger):
    def __init__(self, model_name: str, train_dataset: Dataset, num_labels: int, batch_size: int) -> None:
        super(CurriculumTagger, self).__init__(model_name, train_dataset, num_labels, batch_size)

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "longest", truncation = True, return_offsets_mapping = True, is_split_into_words = True)
        texts = texts.to(self.device)
        try:
            labels = batch["label"]
            labels = labels.to(self.device)
        except KeyError:
            return texts["input_ids"], texts["attention_mask"], None
        
        return texts["input_ids"], texts["attention_mask"], labels

    # This method must also return the normalized logits (i.e. the classification probabilities)
    # Be aware that this method receives an already tokenized input
    def forward(self, input_ids, attention_mask, labels) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        out = self.transformer(input_ids, attention_mask).last_hidden_state
        out = self.linear(out)
        preds = torch.argmax(out, dim = -1)
        probs = torch.max(out, dim = -1)
        if labels:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1), ignore_index = -100)
        else:
            loss = None
        
        return preds, probs, loss


class NERCurriculumLearner(CurriculumLearningModel):
    def __init__(self, student_model: CurriculumTagger, teacher_model: DifficultyPredictor, train_dataloader_args: Dict[str, Any], learning_rate: float,
                 logging : bool, to_string_labels : Callable[[int], str]) -> None:
        super(CurriculumLearningModel, self).__init__(student_model, teacher_model, train_dataloader_args, learning_rate, logging)
        self.to_string_labels = to_string_labels

    def compute_per_sample_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        batch_size = input_ids.size(0)
        with torch.no_grad():
            logits = self.student_model.linear(self.student_model.transformer(input_ids, attention_mask = attention_mask).last_hidden_state)
            per_sample_loss = []
            for i in range(batch_size):
                sample_logits = logits[i].unsqueeze(0)  # [1, T, num_labels]
                sample_labels = labels[i].unsqueeze(0)  # [1, T]
                loss = F.cross_entropy(sample_logits.view(-1, sample_logits.size(-1)), sample_labels.view(-1), ignore_index = -100, reduction = 'mean')
                per_sample_loss.append(loss.item())
            
            target_losses = torch.tensor(per_sample_loss, device = self.device)

        return target_losses
    
    def compute_total_loss(self, losses) -> torch.Tensor:
        return losses
    
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().tolist()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().tolist()

        # Filter out -100 (ignore tokens)
        true_labels = []
        pred_labels = []
        
        for pred_seq, true_seq in zip(y_pred, y_true):
            filtered_true = []
            filtered_pred = []
            for p, t in zip(pred_seq, true_seq):
                if t != -100:  # Skip ignored tokens
                    filtered_true.append(self.to_string_labels(int(t)))
                    filtered_pred.append(self.to_string_labels(int(p)))
            
            if filtered_true:  # Only add non-empty sequences
                true_labels.append(filtered_true)
                pred_labels.append(filtered_pred)

        if not true_labels:
            return

        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")
        recall = recall_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")
        f1 = f1_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")

        batch_size = len(true_labels)
        with warnings.catch_warnings(action = "ignore"):
            self.log(f"{step_type} accuracy", acc, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} precision", precision, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} recall", recall, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} F1 score", f1, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
    

class FakeNewsTeacherModel(DifficultyPredictor):
    def __init__(self, model_name: str) -> None:
        super(DifficultyPredictor, self).__init__(model_name)

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "longest", truncation = True, return_offsets_mapping = True, is_split_into_words = True)
        texts = texts.to(self.device)
        try:
            labels = batch["label"]
            labels = labels.to(self.device)
        except KeyError:
            return texts["input_ids"], texts["attention_mask"]
        
        return texts["input_ids"], texts["attention_mask"], labels


class FakeNewsStudentModel(CurriculumTagger):
    def __init__(self, model_name: str, train_dataset: Dataset, num_labels: int, batch_size: int) -> None:
        super(CurriculumTagger, self).__init__(model_name, train_dataset, num_labels, batch_size)

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "longest", truncation = True, return_offsets_mapping = True, is_split_into_words = True)
        texts = texts.to(self.device)
        try:
            labels = batch["label"]
            labels = labels.to(self.device)
        except KeyError:
            return texts["input_ids"], texts["attention_mask"], None
        
        return texts["input_ids"], texts["attention_mask"], labels

    # This method must also return the normalized logits (i.e. the classification probabilities)
    # Be aware that this method receives an already tokenized input
    def forward(self, input_ids, attention_mask, labels) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        out = self.transformer(input_ids, attention_mask).last_hidden_state
        out = self.linear(out)
        preds = torch.argmax(out, dim = -1)
        probs = torch.max(out, dim = -1)
        if labels:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1), ignore_index = -100)
        else:
            loss = None
        
        return preds, probs, loss


class FakeNewsCurriculumLearner(CurriculumLearningModel):
    def __init__(self, student_model: CurriculumTagger, teacher_model: DifficultyPredictor, train_dataloader_args: Dict[str, Any], learning_rate: float,
                 logging : bool) -> None:
        super(CurriculumLearningModel, self).__init__(student_model, teacher_model, train_dataloader_args, learning_rate, logging)

    def compute_per_sample_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        batch_size = input_ids.size(0)
        with torch.no_grad():
            logits = self.student_model.linear(self.student_model.transformer(input_ids, attention_mask = attention_mask).last_hidden_state)
            per_sample_loss = []
            for i in range(batch_size):
                sample_logits = logits[i].unsqueeze(0)  # [1, T, num_labels]
                sample_labels = labels[i].unsqueeze(0)  # [1, T]
                loss = F.cross_entropy(sample_logits.view(-1, sample_logits.size(-1)), sample_labels.view(-1), ignore_index = -100, reduction = 'none')
                per_sample_loss.append(loss.item())
            
            target_losses = torch.tensor(per_sample_loss, device = self.device)

        return target_losses
    
    def compute_total_loss(self, losses) -> torch.Tensor:
        return losses
    
    def compute_and_log_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, step_type: str) -> None:
        y_pred = y_pred.detach()
        y_true = y_true.detach()

        accuracy = Accuracy(task = "multiclass", num_classes = self.num_labels, average = "macro").to(self.device)
        precision = Precision(task = "multiclass", num_classes = self.num_labels, average = "macro").to(self.device)
        recall = Recall(task = "multiclass", num_classes = self.num_labels, average = "macro").to(self.device)
        f1_score = F1Score(task = "multiclass", num_classes = self.num_labels, average = "macro").to(self.device)

        acc = accuracy(y_pred, y_true)
        prec = precision(y_pred, y_true)
        rec = recall(y_pred, y_true)
        f1 = f1_score(y_pred, y_true)

        with warnings.catch_warnings(action = "ignore"):
            self.log(f"{step_type} accuracy", acc, prog_bar = True, batch_size = self.batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} precision", prec, prog_bar = True, batch_size = self.batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} recall", rec, prog_bar = True, batch_size = self.batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} F1 score", f1, prog_bar = True, batch_size = self.batch_size, on_step = False, on_epoch = True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = "Task choice. Available options: fake_news, legal_ner")
    
    fake_news_parser = subparsers.add_parser("fake_news")
    fake_news_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    fake_news_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    fake_news_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    fake_news_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    fake_news_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-cased-v1", help = "Name of pretrained model to be used")
    fake_news_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    fake_news_parser.set_defaults(func = perform_fake_news_experiments)
    
    legal_nero_parser = subparsers.add_parser("legal_ner")
    legal_nero_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    legal_nero_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    legal_nero_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    legal_nero_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    legal_nero_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-ner", help = "Name of pretrained model to be used")
    legal_nero_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    legal_nero_parser.set_defaults(func = perform_legal_nero_experiments)
    
    args = parser.parse_args()
    return args

def get_datasets(g : torch.Generator, args : argparse.Namespace, task : str) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any], int]:
    match task:
        case "NER":
            training_data, validation_data, test_data, _ = read_legal_ner_dataset()
            training_dataset, validation_dataset, test_dataset = [MarcellDataset(x) for x in [training_data, validation_data, test_data]]
            num_labels = MarcellDataset.num_labels
            collator = MarcellDataset.collator
        case "fake_news":
            training_data, validation_data, test_data, _ = read_fake_news_dataset()
            training_dataset, validation_dataset, test_dataset = [FakeNewsDataset(x) for x in [training_data, validation_data, test_data]]
            num_labels = FakeNewsDataset.num_labels
            collator = FakeNewsDataset.collator
        case _:
            raise NotImplementedError("This task is not supported.")

    training_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": True,
        "pin_memory": True,
        "collate_fn": collator,
        "generator": g
    }

    return training_dataset, validation_dataset, test_dataset, training_args, num_labels

def perform_experiments(g : torch.Generator, args : argparse.Namespace, task : str) -> None:
    training_dataset, validation_dataset, test_dataset, training_args, num_labels = get_datasets(g, args, task)

    train_dataloader = DataLoader(training_dataset, shuffle = True, **training_args)
    validation_dataloader = DataLoader(validation_dataset, shuffle = False, **training_args)
    test_dataloader = DataLoader(test_dataset, shuffle = False, **training_args)

    match task:
        case "NER":
            teacher_model = NERTeacherModel(args.model_name)
            student_model = NERStudentModel(args.model_name, training_dataset, num_labels, args.batch_size)
            model = NERCurriculumLearner(student_model, teacher_model, training_args, args.lr, args.logging, MarcellDataset.to_string_label)
            task_name = "NER"
        case "fake_news":
            teacher_model = FakeNewsTeacherModel(args.model_name)
            student_model = FakeNewsStudentModel(args.model_name, training_dataset, num_labels, args.batch_size)
            model = FakeNewsCurriculumLearner(student_model, teacher_model, training_args, args.lr, args.logging)
            task_name = "fake news"
    
    aim_logger = AimLogger(f"aim_logs_curriculum_{task}", f"Curriculum learning for {task_name}", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.max_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.validate(model, validation_dataloader)
    trainer.test(model, test_dataloader)

perform_legal_nero_experiments = partial(perform_experiments, task = "NER")
perform_fake_news_experiments = partial(perform_experiments, task = "fake_news")

def main() -> None:
    pl.seed_everything(42, workers = True) # 42 is the seed value
    g = torch.Generator()
    g.manual_seed(42)

    args = get_args()
    run_experiments = args.func
    run_experiments(g, args)

if __name__ == "__main__":
    main()