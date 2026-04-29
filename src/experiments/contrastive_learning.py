import argparse
import warnings
import torch
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

from functools import partial
from typing import List, Tuple, Dict, Optional, Callable, Any

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from seqeval.scheme import IOB2

from . import MarcellDataset, FakeNewsDataset
from . import read_legal_ner_dataset, read_fake_news_dataset

from . import ContrastiveNetwork

from . import MyCallback

class NERContrastive(ContrastiveNetwork):
    def __init__(self, model_name: str, num_labels: int, batch_size: int, learning_rate: float, contrastive_weight: float, temperature: float,
                 logging : bool, to_string_labels : Callable[[int], str]) -> None:
        super().__init__(model_name, num_labels, batch_size, learning_rate, contrastive_weight, temperature, logging)
        self.to_string_labels = to_string_labels

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512, return_offsets_mapping = True)
        texts = texts.to(self.device)
        if "label" in batch:
            labels = batch["label"]
            max_len = texts["input_ids"].shape[1]
            labels = pad_sequence([torch.tensor(label, dtype = torch.long) for label in labels], batch_first = True, padding_value = -100).to(self.device)
            if labels.shape[1] > max_len:
                labels = labels[:, :max_len]
            elif labels.shape[1] < max_len:
                padding = torch.full((labels.shape[0], max_len - labels.shape[1]), -100, dtype = torch.long)
                labels = torch.cat([labels, padding], dim = 1).to(self.device)
        else:
            return texts["input_ids"], texts["attention_mask"], None
        
        return texts["input_ids"], texts["attention_mask"], labels

    def forward(self, batch : Dict[str, torch.Tensor], compute_contrastive = True) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        
        transformer_out = self.transformer(input_ids, attention_mask).last_hidden_state
        out = self.linear(transformer_out)
        preds = torch.argmax(out, dim = -1)
        probs = torch.softmax(out, dim = -1)
        if labels is None:
            return preds, probs, (None, None)
        
        ce_loss = self.cross_entropy_loss(out.reshape(-1, self.num_labels), labels.reshape(-1))
        contrastive_loss = None
        if compute_contrastive and self.contrastive_weight > 0.0:
            valid_mask = labels != -100
            if valid_mask.sum() > 1:
                # we need to flatten everything first...
                flat_embeddings = transformer_out.reshape(-1, transformer_out.size(-1))
                flat_labels = labels.reshape(-1)
                flat_mask = valid_mask.reshape(-1)  # ...including the mask
                
                valid_embeddings = flat_embeddings[flat_mask]
                valid_labels = flat_labels[flat_mask]
                
                contrastive_loss = self.contrastive_loss(valid_embeddings, valid_labels, self.linear[1].weight)

        return preds, probs, (ce_loss, contrastive_loss)

    def filter_out_padding_tokens(self, y_pred, y_true) -> Tuple[List[List[str]], List[List[str]]]:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().tolist()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().tolist()

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
            return [], [] 
        
        return pred_labels, true_labels
    
    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        pred_labels, true_labels = self.filter_out_padding_tokens(y_pred, y_true)
        if true_labels == []:
            return

        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")
        recall = recall_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")
        f1 = f1_score(true_labels, pred_labels, zero_division = 0, mode = "strict", scheme = IOB2, average = "macro")

        if self.logging:
            batch_size = self.batch_size # len(true_labels)
            with warnings.catch_warnings(action = "ignore"):
                self.log(f"{step_type} accuracy", acc, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
                self.log(f"{step_type} precision", precision, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
                self.log(f"{step_type} recall", recall, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
                self.log(f"{step_type} F1 score", f1, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)


class FakeNewsContrastive(ContrastiveNetwork):
    def __init__(self, model_name: str, num_labels: int, batch_size: int, learning_rate: float, contrastive_weight: float, temperature: float,
                 logging : bool) -> None:
        super().__init__(model_name, num_labels, batch_size, learning_rate, contrastive_weight, temperature, logging)
        
        metrics = MetricCollection({
            "accuracy": Accuracy(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "precision": Precision(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "recall": Recall(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "f1_score": F1Score(task = "multiclass", num_classes = self.num_labels, average = "macro")
        })
        self.train_metrics = metrics.clone(prefix = "Training ")
        self.validation_metrics = metrics.clone(prefix = "Validation ")
        self.test_metrics = metrics.clone(prefix = "Test ")

    def tokenize_input(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        texts = batch["text"]
        texts = self.tokenizer(texts, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512, return_offsets_mapping = True)
        texts = texts.to(self.device)
        if "label" in batch:
            labels = batch["label"]
            if len(labels) > 0:
                labels = torch.tensor(labels, dtype = torch.long, device = self.device)
        else:
            return texts["input_ids"], texts["attention_mask"], None
        
        return texts["input_ids"], texts["attention_mask"], labels

    def forward(self, batch : Dict[str, torch.Tensor], compute_contrastive = True) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        
        transformer_out = self.transformer(input_ids, attention_mask).last_hidden_state
        cls_embeddings = transformer_out[:, 0, :] # get the embeddings for the [CLS] token (only these are necessary for the linear layer)
        out = self.linear(cls_embeddings)
        preds = torch.argmax(out, dim = -1)
        probs = torch.softmax(out, dim = -1)
        if labels is None:
            return preds, probs, (None, None)
        
        ce_loss = self.cross_entropy_loss(out, labels)
        contrastive_loss = None
        if compute_contrastive and self.contrastive_weight > 0.0:
            contrastive_loss = self.contrastive_loss(cls_embeddings, labels, self.linear[1].weight)

        return preds, probs, (ce_loss, contrastive_loss)

    def compute_and_log_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, step_type: str) -> None:
        if len(y_true) == 0:
            return
        
        y_pred = y_pred.detach().float()
        y_true = y_true.detach().long()
        mapping = {
            "Training": self.train_metrics,
            "Validation": self.validation_metrics,
            "Test": self.test_metrics
        }
        metric_collection = mapping.get(step_type)
        output = metric_collection(y_pred, y_true)

        if self.logging:
            with warnings.catch_warnings(action = "ignore"):
                self.log_dict(output, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)


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
    fake_news_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    fake_news_parser.add_argument("--temperature", type = float, default = 0.07, help = "Temperature")
    fake_news_parser.set_defaults(func = perform_fake_news_experiments)
    
    legal_nero_parser = subparsers.add_parser("legal_ner")
    legal_nero_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    legal_nero_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    legal_nero_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    legal_nero_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    legal_nero_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-ner", help = "Name of pretrained model to be used")
    legal_nero_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    legal_nero_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    legal_nero_parser.add_argument("--temperature", type = float, default = 0.07, help = "Temperature")
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
            model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                   MarcellDataset.to_string_label)
            task_name = "NER"
        case "fake_news":
            model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging)
            task_name = "fake news"
        case _:
            raise NotImplementedError("This task is not implemented.")
    
    aim_logger = AimLogger(f"aim_logs_contrastive_{task}", f"Contrastive learning for {task_name}", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.num_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
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