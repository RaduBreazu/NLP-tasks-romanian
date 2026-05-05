import argparse
import warnings
import torch
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from copy import deepcopy

from typing import List, Dict, Tuple, Any

from torch.utils.data import Dataset, DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from seqeval.scheme import IOB2

from . import FakeNewsDataset, read_fake_news_dataset
from . import MarcellDataset, read_legal_ner_dataset

from . import VotingMechanism, EnsembleModel

from . import MyCallback

from .plain import NERPlainTagger, FakeNewsPlainDetector
from .curriculum_learning import NERTeacherModel, NERStudentModel, NERCurriculumLearner, FakeNewsTeacherModel, FakeNewsStudentModel, FakeNewsCurriculumLearner
from .contrastive_learning import NERContrastive, FakeNewsContrastive
from .curriculum_and_contrastive import NERCurriculumContrastiveLearner, FakeNewsCurriculumContrastiveLearner

class LinearVoting(VotingMechanism):
    def __init__(self, num_models : int) -> None:
        super().__init__()
        n = num_models
        self.weights = torch.full((n,), 1.0 / n, dtype = torch.float32, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __call__(self, probs : torch.Tensor, losses : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dims = probs.dim()
        view_shape = [-1] + ([1] * (num_dims - 1))
        w_probs = self.weights.view(*view_shape)
        new_probs = (probs * w_probs).sum(dim = 0)
        new_preds = torch.argmax(new_probs, dim = -1)
        confidence_scores = torch.max(new_probs, dim = -1).values
        if losses.numel() == 0:
            return new_preds, confidence_scores, None
            
        w_losses = self.weights.view(-1, 1) if losses.dim() == 2 else self.weights
        new_losses = (losses * w_losses).sum(dim = 0)
        new_scalar_loss = new_losses.mean() if new_losses.dim() > 0 else new_losses
        return new_preds, confidence_scores, new_scalar_loss


class NEREnsemble(EnsembleModel):
    def __init__(self, models: List[pl.LightningModule], batch_size : int, num_labels : int, voting_mechanism : VotingMechanism, learning_rate: float, logging : bool) -> None:
        super().__init__(models, batch_size, num_labels, voting_mechanism, learning_rate, logging)

    def filter_out_padding_tokens(self, y_pred, y_true) -> Tuple[List[List[str]], List[List[str]]]:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().tolist()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().tolist()

        to_string_labels = self.models[0].to_string_labels # all models have the same `to_string_labels` implementation, since they are all trained on the same dataset
        
        true_labels = []
        pred_labels = []
        
        for pred_seq, true_seq in zip(y_pred, y_true):
            filtered_true = []
            filtered_pred = []
            for p, t in zip(pred_seq, true_seq):
                if t != -100:
                    filtered_true.append(to_string_labels(int(t)))
                    filtered_pred.append(to_string_labels(int(p)))
            
            if filtered_true:
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

        batch_size = self.batch_size # len(true_labels)
        with warnings.catch_warnings(action = "ignore"):
            self.log(f"{step_type} accuracy", acc, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} precision", precision, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} recall", recall, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} F1 score", f1, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
    

class FakeNewsEnsemble(EnsembleModel):
    def __init__(self, models: List[pl.LightningModule], batch_size : int, num_labels : int, voting_mechanism : VotingMechanism, learning_rate: float, logging : bool) -> None:
        super().__init__(models, batch_size, num_labels, voting_mechanism, learning_rate, logging)

        metrics = MetricCollection({
            "accuracy": Accuracy(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "precision": Precision(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "recall": Recall(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "f1_score": F1Score(task = "multiclass", num_classes = self.num_labels, average = "macro")
        })
        self.train_metrics = metrics.clone(prefix = "Training ")
        self.validation_metrics = metrics.clone(prefix = "Validation ")
        self.test_metrics = metrics.clone(prefix = "Test ")

    def compute_and_log_metrics(self, y_pred, y_true, step_type: str) -> None:
        if len(y_true) == 0:
            return
        
        match y_pred:
            case list([*ys]):
                y_pred = torch.tensor(ys, dtype = torch.float32, device = self.device)
            case torch.Tensor:
                y_pred = y_pred.detach().float()

        match y_true:
            case list([*ys]):
                y_true = torch.tensor(ys, dtype = torch.long, device = self.device)
            case torch.Tensor:
                y_true = y_true.detach().long()
        
        mapping = {
            "Training": self.train_metrics,
            "Validation": self.validation_metrics,
            "Test": self.test_metrics
        }
        metric_collection = mapping.get(step_type)
        output = metric_collection(y_pred, y_true)

        with warnings.catch_warnings(action = "ignore"):
            self.log_dict(output, prog_bar = True, on_step = False, on_epoch = True, batch_size = self.batch_size)


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

    def generate_curriculum_learner(type : str, model_name : str) -> NERCurriculumLearner | FakeNewsCurriculumLearner:
        match type:
            case "NER":
                curriculum_teacher_model = NERTeacherModel(model_name)
                curriculum_student_model = NERStudentModel(model_name, training_dataset, num_labels, args.batch_size)
                curriculum_learner = NERCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, args.lr, False,
                                                          MarcellDataset.to_string_label)
            case "fake_news":
                curriculum_teacher_model = FakeNewsTeacherModel(model_name)
                curriculum_student_model = FakeNewsStudentModel(model_name, training_dataset, num_labels, args.batch_size)
                curriculum_learner = FakeNewsCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, args.lr, False)
        
        return curriculum_learner
    
    match args.experiment:
        case "ner_simple_models":
            model_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            models = [NERPlainTagger(name, MarcellDataset.num_labels, args.batch_size, args.lr, False, MarcellDataset.to_string_label) for name in model_names]
            proper_task_name = "NER"
        case "ner_curriculum":
            model_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            models = [generate_curriculum_learner("NER", name) for name in model_names]
            proper_task_name = "NER"
        case "ner_contrastive":
            model_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            models = [NERContrastive(name, MarcellDataset.num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, False,
                                     MarcellDataset.to_string_label) for name in model_names]
            proper_task_name = "NER"
        case "ner_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner("NER", args.model_name)
            contrastive_learner = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                                 MarcellDataset.to_string_label)
            curriculum_contrastive_learner = NERCurriculumContrastiveLearner(deepcopy(contrastive_learner), deepcopy(curriculum_learner.teacher_model), training_args,
                                                                             args.lr, False, MarcellDataset.collator)
            models = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            proper_task_name = "NER"
        case "fake_news_simple_models":
            model_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            models = [FakeNewsPlainDetector(name, FakeNewsDataset.num_labels, args.batch_size, args.lr, logging = False) for name in model_names]
            proper_task_name = "fake news detection"
        case "fake_news_curriculum":
            model_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            models = [generate_curriculum_learner("fake_news", name) for name in model_names]
            proper_task_name = "fake news detection"
        case "fake_news_contrastive":
            model_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            models = [FakeNewsContrastive(name, FakeNewsDataset.num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, logging = False)
                      for name in model_names]
            proper_task_name = "fake_news"
        case "fake_news_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner("fake_news", args.model_name)
            contrastive_learner = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, logging = False)
            curriculum_contrastive_learner = FakeNewsCurriculumContrastiveLearner(deepcopy(curriculum_learner.student_model), deepcopy(curriculum_learner.teacher_model),
                                                                                  training_args, args.lr, logging = False)
            models = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            proper_task_name = "fake news detection"
    
    voting_mechanism = LinearVoting(len(models))
    match task:
        case "NER":
            model = NEREnsemble(models, args.batch_size, num_labels, voting_mechanism, args.lr, args.logging)
        case "fake_news":
            model = FakeNewsEnsemble(models, args.batch_size, num_labels, voting_mechanism, args.lr, args.logging)
        
    aim_logger = AimLogger(f"aim_logs_ensemble_{task}", f"Ensemble learning for {proper_task_name}", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.num_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.validate(model, validation_dataloader)
    trainer.test(model, test_dataloader)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = "Task choice. Available options: fake_news, legal_ner")
    
    fake_news_parser = subparsers.add_parser("fake_news")
    fake_news_parser.add_argument("--num_epochs", type = int, default = 20, help = "Number of epochs")
    fake_news_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    fake_news_parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    fake_news_parser.add_argument("--batch_size", type = int, default = 16, help = "Batch size")
    fake_news_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-cased-v1", help = "Name of pretrained model to be used")
    fake_news_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    fake_news_parser.add_argument("--temperature", type = float, default = 0.1, help = "Temperature")
    fake_news_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    fake_news_parser.add_argument("--experiment", type = str, default = "fake_news_simple_models", help = "Experiment name",
                                  choices = ["fake_news_simple_models", "fake_news_curriculum", "fake_news_contrastive", "fake_news_curriculum_contrastive"])
    
    legal_nero_parser = subparsers.add_parser("legal_ner")
    legal_nero_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    legal_nero_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    legal_nero_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    legal_nero_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    legal_nero_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-ner", help = "Name of pretrained model to be used")
    legal_nero_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    legal_nero_parser.add_argument("--temperature", type = float, default = 0.07, help = "Temperature")
    legal_nero_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    legal_nero_parser.add_argument("--experiment", type = str, default = "ner_simple_models", help = "Experiment name",
                                  choices = ["ner_simple_models", "ner_curriculum", "ner_contrastive", "ner_curriculum_contrastive"])
    
    args = parser.parse_args()
    return args

def main() -> None:
    pl.seed_everything(42, workers = True) # 42 is the seed value
    g = torch.Generator()
    g.manual_seed(42)

    args = get_args()
    match args.experiment:
        case str(x) if x.startswith("ner"):
            task_name = "NER"
        case str(x) if x.startswith("fake_news"):
            task_name = "fake_news"
    
    perform_experiments(g, args, task_name)

if __name__ == "__main__":
    main()