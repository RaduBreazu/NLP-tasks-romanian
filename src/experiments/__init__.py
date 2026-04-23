import os
import gc
import torch
import pytorch_lightning as pl

from utils.fake_news import FakeNewsDataset, read_fake_news_dataset
from utils.marcell import MarcellDataset
from utils.juro import JuRODataset, read_legal_ner_dataset

from methods.plain import PlainNetwork
from methods.contrastive_learning import ContrastiveNetwork
from methods.curriculum_learning import CurriculumTagger, CurriculumLearningModel
from methods.mixture import VotingMechanism, MixtureModel
from methods.self_training import SelfTrainingModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")

class MyCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()