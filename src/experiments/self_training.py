import argparse
import torch
import pytorch_lightning as pl
from aim.pytorch_lightning import AimLogger
from copy import deepcopy

from torch.utils.data import DataLoader

from . import FakeNewsDataset, read_fake_news_dataset
from . import MarcellDataset, JuRODataset, read_legal_ner_dataset

from . import SelfTrainingModel

from . import MyCallback

from plain import NERPlainTagger, FakeNewsPlainDetector
from curriculum_learning import NERTeacherModel, NERStudentModel, NERCurriculumLearner, FakeNewsTeacherModel, FakeNewsStudentModel, FakeNewsCurriculumLearner
from contrastive_learning import NERContrastive, FakeNewsContrastive
from curriculum_and_contrastive import NERCurriculumContrastiveLearner, FakeNewsCurriculumContrastiveLearner
from mixture import LinearVoting, NERMixture, FakeNewsMixture

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = "Task choice. Available options: fake_news, legal_ner")
    
    fake_news_experiment_list = ["fake_news_simple_model", "fake_news_curriculum", "fake_news_contrastive", "fake_news_curriculum_contrastive",
                                 "fake_news_mixture_simple_models", "fake_news_mixture_curriculum", "fake_news_mixture_contrastive",
                                 "fake_news_mixture_curriculum_contrastive"]
    fake_news_parser = subparsers.add_parser("fake_news")
    fake_news_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    fake_news_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    fake_news_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    fake_news_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    fake_news_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-cased-v1", help = "Name of pretrained model to be used")
    fake_news_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    fake_news_parser.add_argument("--temperature", type = float, default = 0.07, help = "Temperature")
    fake_news_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    fake_news_parser.add_argument("--experiment", type = str, default = "fake_news_simple_model", help = "Experiment name",
                                  choices = fake_news_experiment_list)
    
    ner_experiment_list = ["ner_simple_model", "ner_curriculum", "ner_contrastive", "ner_curriculum_contrastive", "ner_mixture_simple_models", "ner_mixture_curriculum",
                           "ner_mixture_contrastive", "ner_mixture_curriculum_contrastive"]
    legal_nero_parser = subparsers.add_parser("legal_ner")
    legal_nero_parser.add_argument("--num_epochs", type = int, default = 10, help = "Number of epochs")
    legal_nero_parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers")
    legal_nero_parser.add_argument("--lr", type = float, default = 2e-5, help = "Learning rate")
    legal_nero_parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    legal_nero_parser.add_argument("--model_name", type = str, default = "dumitrescustefan/bert-base-romanian-ner", help = "Name of pretrained model to be used")
    legal_nero_parser.add_argument("--contrastive_weight", type = float, default = 0.3, help = "Importance of contrastive loss in total loss")
    legal_nero_parser.add_argument("--temperature", type = float, default = 0.07, help = "Temperature")
    legal_nero_parser.add_argument("--logging", type = bool, default = True, help = "Whether or not to log the results of the experiments")
    legal_nero_parser.add_argument("--experiment", type = str, default = "ner_simple_model", help = "Experiment name",
                                  choices = ner_experiment_list)
    
    args = parser.parse_args()
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

    def generate_curriculum_learner(model_name : str) -> NERCurriculumLearner:
        curriculum_teacher_model = NERTeacherModel(model_name)
        curriculum_student_model = NERStudentModel(model_name, training_dataset, num_labels, args.batch_size)
        curriculum_learner = NERCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, args.lr, args.logging, MarcellDataset.to_string_label)
        return curriculum_learner

    # choose the base model according to the experiment that was provided
    num_labels = MarcellDataset.num_labels
    match args.experiment:
        case "ner_simple_model":
            supervised_model = NERPlainTagger(args.model_name, num_labels, args.batch_size, args.lr, args.logging, MarcellDataset.to_string_label)
            experiment_name = "plain model"
        case "ner_curriculum":
            supervised_model = generate_curriculum_learner(args.model_name)
            experiment_name = "curriculum learning"
        case "ner_contrastive":
            supervised_model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                              MarcellDataset.to_string_label)
            experiment_name = "contrastive learning"
        case "ner_curriculum_contrastive":
            teacher_model = NERTeacherModel(args.model_name)
            student_model = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                           MarcellDataset.to_string_label)
            supervised_model = NERCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr, args.logging, MarcellDataset.to_string_label)
            experiment_name = "curriculum and contrastive learning"
        case "ner_mixture_simple_models":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "umitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [NERPlainTagger(name, num_labels, args.batch_size, args.lr, args.logging, MarcellDataset.to_string_label) for name in expert_names]
            supervised_model = NERMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of plain experts"
        case "ner_mixture_curriculum":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "umitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [generate_curriculum_learner(name) for name in expert_names]
            supervised_model = NERMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of curriculum learning experts"
        case "ner_mixture_contrastive":
            expert_names = ["dumitrescustefan/bert-base-romanian-ner", "dumitrescustefan/bert-base-romanian-ner", "umitrescustefan/bert-base-romanian-uncased-v1"]
            experts = [NERContrastive(name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging, MarcellDataset.to_string_label)
                       for name in expert_names]
            supervised_model = NERMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of contrastive learning experts"
        case "ner_mixture_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner(args.model_name)
            contrastive_learner = NERContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging,
                                                 MarcellDataset.to_string_label)
            curriculum_contrastive_learner = NERCurriculumContrastiveLearner(deepcopy(contrastive_learner), deepcopy(curriculum_learner.teacher_model), training_args,
                                                                             args.lr, args.logging, MarcellDataset.to_string_label)
            experts = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            supervised_model = NERMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of curriculum and contrastive learning experts"

    model = SelfTrainingModel(supervised_model, training_data, unlabelled_dataset)
    aim_logger = AimLogger(f"aim_logs_self_training_{args.experiment}", f"Self-training of {experiment_name} for fake news", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.max_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
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

    def generate_curriculum_learner(model_name : str) -> FakeNewsCurriculumLearner:
        curriculum_teacher_model = FakeNewsTeacherModel(model_name)
        curriculum_student_model = FakeNewsStudentModel(model_name, training_dataset, num_labels, args.batch_size)
        curriculum_learner = FakeNewsCurriculumLearner(curriculum_student_model, curriculum_teacher_model, training_args, args.lr, args.logging)
        return curriculum_learner

    # choose the base model according to the experiment that was provided
    num_labels = len(training_data["label"].unique())
    match args.experiment:
        case "fake_news_simple_model":
            supervised_model = FakeNewsPlainDetector(args.model_name, num_labels, args.batch_size, args.lr, args.logging)
            experiment_name = "plain model"
        case "fake_news_curriculum":
            supervised_model = generate_curriculum_learner(args.model_name)
            experiment_name = "curriculum learning"
        case "fake_news_contrastive":
            supervised_model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging)
            experiment_name = "contrastive learning"
        case "fake_news_curriculum_contrastive":
            teacher_model = FakeNewsTeacherModel(args.model_name)
            student_model = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging)
            supervised_model = FakeNewsCurriculumContrastiveLearner(student_model, teacher_model, training_args, args.lr, args.logging)
            experiment_name = "curriculum and contrastive learning"
        case "fake_news_mixture_simple_models":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [FakeNewsPlainDetector(name, num_labels, args.batch_size, args.lr, args.logging) for name in expert_names]
            supervised_model = FakeNewsMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of plain experts"
        case "fake_news_mixture_curriculum":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [generate_curriculum_learner(name) for name in expert_names]
            supervised_model = FakeNewsMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of curriculum learning experts"
        case "fake_news_mixture_contrastive":
            expert_names = ["dumitrescustefan/bert-base-romanian-cased-v1", "readerbench/RoBERT-base", "readerbench/RoBERT-large"]
            experts = [FakeNewsContrastive(name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging) for name in expert_names]
            supervised_model = FakeNewsMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of contrastive learning experts"
        case "fake_news_mixture_curriculum_contrastive":
            curriculum_learner = generate_curriculum_learner(args.model_name)
            contrastive_learner = FakeNewsContrastive(args.model_name, num_labels, args.batch_size, args.lr, args.contrastive_weight, args.temperature, args.logging)
            curriculum_contrastive_learner = FakeNewsCurriculumContrastiveLearner(deepcopy(contrastive_learner), deepcopy(curriculum_learner.teacher_model),
                                                                                  training_args, args.lr, args.logging)
            experts = [curriculum_learner, contrastive_learner, curriculum_contrastive_learner]
            supervised_model = FakeNewsMixture(experts, args.batch_size, LinearVoting(len(experts)), args.logging)
            experiment_name = "mixture of curriculum and contrastive learning experts"

    model = SelfTrainingModel(supervised_model, training_dataset, unlabelled_dataset)
    aim_logger = AimLogger(f"aim_logs_self_training_{args.experiment}", f"Self-training with {experiment_name} for fake news", "Train_", "Validation_", "Test_")
    trainer = pl.Trainer(max_epochs = args.max_epochs, deterministic = True, logger = aim_logger, log_every_n_steps = 1, callbacks = [MyCallback()])
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