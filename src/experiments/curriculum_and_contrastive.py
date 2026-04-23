import argparse
import torch
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger

from functools import partial
from typing import Tuple, Dict, Callable, Any
from torch.utils.data import Dataset, DataLoader

from contrastive_learning import NERContrastive, FakeNewsContrastive
from curriculum_learning import NERTeacherModel, NERCurriculumLearner, FakeNewsTeacherModel, FakeNewsCurriculumLearner

from . import MarcellDataset, FakeNewsDataset
from . import read_legal_ner_dataset, read_fake_news_dataset

from . import MyCallback

class NERCurriculumContrastiveLearner(NERCurriculumLearner):
    def __init__(self, student_model: NERContrastive, teacher_model: NERTeacherModel, train_dataloader_args: Dict[str, Any], learning_rate: float,
                 logging : bool, to_string_labels : Callable[[int], str]) -> None:
        super().__init__(student_model, teacher_model, train_dataloader_args, learning_rate, logging, to_string_labels)
        self.student_model.logging = logging

    def compute_total_loss(self, losses) -> torch.Tensor:
        ce_loss, cont_loss = losses
        return ce_loss + self.student_model.contrastive_weight * cont_loss
    

class FakeNewsCurriculumContrastiveLearner(FakeNewsCurriculumLearner):
    def __init__(self, student_model: FakeNewsContrastive, teacher_model: FakeNewsTeacherModel, train_dataloader_args: Dict[str, Any], learning_rate: float,
                 logging : bool) -> None:
        super().__init__(student_model, teacher_model, train_dataloader_args, learning_rate, logging)
        self.student_model.logging = logging

    def compute_total_loss(self, losses) -> torch.Tensor:
        ce_loss, cont_loss = losses
        return ce_loss + self.student_model.contrastive_weight * cont_loss


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
            teacher_model = NERTeacherModel(args.model_name)
            student_model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                           MarcellDataset.to_string_label)
            model = NERCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr, args.logging, MarcellDataset.to_string_label)
            task_name = "NER"
        case "fake_news":
            teacher_model = FakeNewsTeacherModel(args.model_name)
            student_model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging)
            model = FakeNewsCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr, args.logging)
            task_name = "fake news"
        case _:
            raise NotImplementedError("This task is not implemented.")
        
    aim_logger = AimLogger(f"aim_logs_curriculum_contrastive_{task}", f"Curriculum + contrastive learning for {task_name}", "Train_", "Validation_", "Test_")
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