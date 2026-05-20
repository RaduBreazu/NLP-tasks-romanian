import warnings
import argparse
import torch
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from copy import deepcopy

from typing import List, Tuple, Callable

from torch.utils.data import Dataset, DataLoader

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from seqeval.scheme import IOB2

from . import FakeNewsDataset, read_fake_news_dataset
from . import MarcellDataset, JuRODataset, read_legal_ner_dataset

from . import SelfTrainingModel

from . import MyCallback

from .plain import NERPlainTagger, FakeNewsPlainDetector
from .curriculum_learning import NERTeacherModel, NERStudentModel, NERCurriculumLearner, FakeNewsTeacherModel, FakeNewsStudentModel, FakeNewsCurriculumLearner
from .contrastive_learning import NERContrastive, FakeNewsContrastive
from .curriculum_and_contrastive import NERCurriculumContrastiveLearner, FakeNewsCurriculumContrastiveLearner
from .ensemble import LinearVoting, NEREnsemble, FakeNewsEnsemble

class NERSelfTraining(SelfTrainingModel):
    def __init__(self, model: pl.LightningModule, labelled_data: Dataset, unlabelled_data: Dataset, inner_epochs: int, logging: bool,
                 to_string_labels: Callable[[int], str]) -> None:
        super().__init__(model, labelled_data, unlabelled_data, inner_epochs, logging)
        self.to_string_labels = to_string_labels

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
                if t != -100:
                    filtered_true.append(self.to_string_labels(int(t)))
                    filtered_pred.append(self.to_string_labels(int(p)))
            
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

        batch_size = self.batch_size
        with warnings.catch_warnings(action = "ignore"):
            self.log(f"{step_type} accuracy", acc, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} precision", precision, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} recall", recall, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)
            self.log(f"{step_type} F1 score", f1, prog_bar = True, batch_size = batch_size, on_step = False, on_epoch = True)


class FakeNewsSelfTraining(SelfTrainingModel):
    def __init__(self, model: pl.LightningModule, labelled_data: Dataset, unlabelled_data: Dataset, inner_epochs: int, logging: bool) -> None:
        super().__init__(model, labelled_data, unlabelled_data, inner_epochs, logging)

        metrics = MetricCollection({
            "accuracy": Accuracy(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "precision": Precision(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "recall": Recall(task = "multiclass", num_classes = self.num_labels, average = "macro"),
            "f1_score": F1Score(task = "multiclass", num_classes = self.num_labels, average = "macro")
        })
        self.train_metrics = metrics.clone(prefix = "Training ")
        self.validation_metrics = metrics.clone(prefix = "Validation ")
        self.test_metrics = metrics.clone(prefix = "Test ")

    def compute_and_log_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, step_type: str) -> None:
        if len(y_true) == 0:
            return
        
        match y_pred:
            case list([*ys]):
                y_pred = torch.as_tensor(ys, dtype = torch.float32, device = self.device)
            case torch.Tensor:
                y_pred = y_pred.detach().float()

        match y_true:
            case list([*ys]):
                y_true = torch.as_tensor(ys, dtype = torch.long, device = self.device)
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

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = "Task choice. Available options: fake_news, legal_ner")
    
    fake_news_experiment_list = ["fake_news_simple_model", "fake_news_curriculum", "fake_news_contrastive", "fake_news_curriculum_contrastive",
                                 "fake_news_ensemble_simple_models", "fake_news_ensemble_curriculum", "fake_news_ensemble_contrastive",
                                 "fake_news_ensemble_curriculum_contrastive"]
    fake_news_parser = subparsers.add_parser("fake_news")
    fake_news_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    fake_news_parser.add_argument("--num_inner_epochs", type = int, default = 4, help = "Number of base model epochs per large model epochs")
    fake_news_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    fake_news_parser.add_argument("--lr", type = str, default = "[1e-4, 5e-5, 2e-5]", help = "Learning rates of the models")
    fake_news_parser.add_argument("--batch_size", type = int, default = 16, help = "Batch size")
    fake_news_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-cased-v1", help = "Name of pretrained model to be used")
    fake_news_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    fake_news_parser.add_argument("--temperature", type = float, default = 0.15, help = "Temperature")
    fake_news_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    fake_news_parser.add_argument("--experiment", type = str, default = "fake_news_simple_model", help = "Experiment name",
                                  choices = fake_news_experiment_list)
    
    ner_experiment_list = ["ner_simple_model", "ner_curriculum", "ner_contrastive", "ner_curriculum_contrastive", "ner_ensemble_simple_models", "ner_ensemble_curriculum",
                           "ner_ensemble_contrastive", "ner_ensemble_curriculum_contrastive"]
    legal_nero_parser = subparsers.add_parser("legal_ner")
    legal_nero_parser.add_argument("--num_epochs", type = int, default = 5, help = "Number of epochs")
    legal_nero_parser.add_argument("--num_inner_epochs", type = int, default = 4, help = "Number of base model epochs per large model epochs")
    legal_nero_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    legal_nero_parser.add_argument("--lr", type = str, default = "[2e-5, 2e-5, 2e-5]", help = "Learning rates of the models")
    legal_nero_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    legal_nero_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-ner", help = "Name of pretrained model to be used")
    legal_nero_parser.add_argument("--contrastive_weight", type = float, default = 0.2, help = "Importance of contrastive loss in total loss")
    legal_nero_parser.add_argument("--temperature", type = float, default = 0.05, help = "Temperature")
    legal_nero_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    legal_nero_parser.add_argument("--experiment", type = str, default = "ner_simple_model", help = "Experiment name",
                                  choices = ner_experiment_list)
    
    args = parser.parse_args()
    args.lr = [float(x) for x in args.lr[1:-1].split(", ")]
    return args

def perform_legal_nero_experiments(g : torch.Generator, args : argparse.Namespace) -> None:
    training_data, validation_data, test_data, unlabelled_data = read_legal_ner_dataset()
    training_dataset, validation_dataset, test_dataset = [MarcellDataset(x) for x in [training_data, validation_data, test_data]]
    unlabelled_dataset = JuRODataset(unlabelled_data)

    training_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": True,
        "pin_memory": True,
        "collate_fn": MarcellDataset.collator,
        "generator": g
    }

    train_dataloader = DataLoader(training_dataset, shuffle = True, **training_args)
    validation_dataloader = DataLoader(validation_dataset, shuffle = False, **training_args)
    test_dataloader = DataLoader(test_dataset, shuffle = False, **training_args)

    def generate_curriculum_learner(model_name : str, lr : float) -> NERCurriculumLearner:
        curriculum_teacher_model = NERTeacherModel(model_name)
        curriculum_student_model = NERStudentModel(model_name, training_dataset, num_labels, args.batch_size)
        curriculum_learner = NERCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, lr, False, MarcellDataset.to_string_label)
        return curriculum_learner

    # choose the base model according to the experiment that was provided
    num_labels = MarcellDataset.num_labels
    match args.experiment:
        case "ner_simple_model":
            supervised_model = NERPlainTagger(args.model_name, num_labels, args.batch_size, args.lr[0], False, MarcellDataset.to_string_label)
            experiment_name = "plain model"
        case "ner_curriculum":
            supervised_model = generate_curriculum_learner(args.model_name, args.lr[0])
            experiment_name = "curriculum learning"
        case "ner_contrastive":
            supervised_model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, False,
                                              MarcellDataset.to_string_label)
            experiment_name = "contrastive learning"
        case "ner_curriculum_contrastive":
            teacher_model = NERTeacherModel(args.model_name)
            student_model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, False,
                                           MarcellDataset.to_string_label)
            supervised_model = NERCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr[0], False, MarcellDataset.to_string_label)
            experiment_name = "curriculum and contrastive learning"
        case "ner_ensemble_simple_models":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [NERPlainTagger(name, num_labels, args.batch_size, lr, False, MarcellDataset.to_string_label) for name, lr in zip(expert_names, args.lr)]
            supervised_model = NEREnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of plain experts"
        case "ner_ensemble_curriculum":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [generate_curriculum_learner(name, lr) for name, lr in zip(expert_names, args.lr)]
            supervised_model = NEREnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of curriculum learning experts"
        case "ner_ensemble_contrastive":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [NERContrastive(name, num_labels, args.batch_size, lr, args.contrastive_weight, args.temperature, False, MarcellDataset.to_string_label)
                       for name, lr in zip(expert_names, args.lr)]
            supervised_model = NEREnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of contrastive learning experts"
        case "ner_ensemble_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner(args.model_name, args.lr[0])
            contrastive_learner = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, False,
                                                 MarcellDataset.to_string_label)
            curriculum_contrastive_learner = NERCurriculumContrastiveLearner(deepcopy(contrastive_learner), deepcopy(curriculum_learner.teacher_model), training_args,
                                                                             args.lr[0], False, MarcellDataset.to_string_label)
            experts = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            supervised_model = NEREnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of curriculum and contrastive learning experts"

    model = NERSelfTraining(supervised_model, training_data, unlabelled_dataset, args.num_inner_epochs, args.logging, MarcellDataset.to_string_label)
    aim_logger = AimLogger(f"aim_logs_self_training_{args.experiment}", f"Self-training of {experiment_name} for NER", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.num_epochs * args.num_inner_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.validate(model, validation_dataloader)
    trainer.test(model, test_dataloader)

def perform_fake_news_experiments(g : torch.Generator, args : argparse.Namespace) -> None:
    training_data, validation_data, test_data, unlabelled_data = read_fake_news_dataset()
    training_dataset, validation_dataset, test_dataset, unlabelled_dataset = [FakeNewsDataset(x) for x in [training_data, validation_data, test_data, unlabelled_data]]

    training_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": True,
        "pin_memory": True,
        "collate_fn": FakeNewsDataset.collator,
        "generator": g
    }

    train_dataloader = DataLoader(training_dataset, shuffle = True, **training_args)
    validation_dataloader = DataLoader(validation_dataset, shuffle = False, **training_args)
    test_dataloader = DataLoader(test_dataset, shuffle = False, **training_args)

    def generate_curriculum_learner(model_name : str, lr : float) -> FakeNewsCurriculumLearner:
        curriculum_teacher_model = FakeNewsTeacherModel(model_name)
        curriculum_student_model = FakeNewsStudentModel(model_name, training_dataset, num_labels, args.batch_size)
        curriculum_learner = FakeNewsCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, lr, logging = False)
        return curriculum_learner

    # choose the base model according to the experiment that was provided
    num_labels = len(training_data["label"].unique())
    match args.experiment:
        case "fake_news_simple_model":
            supervised_model = FakeNewsPlainDetector(args.model_name, num_labels, args.batch_size, args.lr[0], logging = False)
            experiment_name = "plain model"
        case "fake_news_curriculum":
            supervised_model = generate_curriculum_learner(args.model_name, args.lr[0])
            experiment_name = "curriculum learning"
        case "fake_news_contrastive":
            supervised_model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, logging = False)
            experiment_name = "contrastive learning"
        case "fake_news_curriculum_contrastive":
            teacher_model = FakeNewsTeacherModel(args.model_name)
            student_model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, logging = False)
            supervised_model = FakeNewsCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr[0], logging = False)
            experiment_name = "curriculum and contrastive learning"
        case "fake_news_ensemble_simple_models":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [FakeNewsPlainDetector(name, num_labels, args.batch_size, lr, logging = False) for name, lr in zip(expert_names, args.lr)]
            supervised_model = FakeNewsEnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of plain experts"
        case "fake_news_ensemble_curriculum":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [generate_curriculum_learner(name, lr) for name, lr in zip(expert_names, args.lr)]
            supervised_model = FakeNewsEnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of curriculum learning experts"
        case "fake_news_ensemble_contrastive":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [FakeNewsContrastive(name, num_labels, args.batch_size, lr, args.contrastive_weight, args.temperature, logging = False) for name, lr in zip(expert_names, args.lr)]
            supervised_model = FakeNewsEnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of contrastive learning experts"
        case "fake_news_ensemble_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner(args.model_name, args.lr[0])
            contrastive_learner = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr[0], args.contrastive_weight, args.temperature, logging = False)
            curriculum_contrastive_learner = FakeNewsCurriculumContrastiveLearner(deepcopy(contrastive_learner), deepcopy(curriculum_learner.teacher_model),
                                                                                  training_args, args.lr[0], logging = False)
            experts = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            supervised_model = FakeNewsEnsemble(experts, args.batch_size, num_labels, LinearVoting(len(experts)), logging = False)
            experiment_name = "ensemble of curriculum and contrastive learning experts"

    model = FakeNewsSelfTraining(supervised_model, training_dataset, unlabelled_dataset, args.num_inner_epochs, args.logging)
    aim_logger = AimLogger(f"aim_logs_self_training_{args.experiment}", f"Self-training with {experiment_name} for fake news", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.num_epochs * args.num_inner_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.validate(model, validation_dataloader)
    trainer.test(model, test_dataloader)

def main() -> None:
    pl.seed_everything(42, workers = True) # 42 is the seed value
    g = torch.Generator()
    g.manual_seed(42)

    args = get_args()
    match args.experiment:
        case str(x) if x.startswith("ner"):
            perform_experiments = perform_legal_nero_experiments
        case str(x) if x.startswith("fake_news"):
            perform_experiments = perform_fake_news_experiments
    
    perform_experiments(g, args)

if __name__ == "__main__":
    main()