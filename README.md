# NLP Tasks on Romanian

## Description

This repository contains the code for several research experiments regarding named entity recognition (NER) and fake news detection on Romanian-language texts I 
have performed. The baseline architecture for all models is BERT, and I have tested both supervised and semi-supervised techniques. The supervised techniques I 
have tested are curriculum learning, contrastive learning and ensemble learning. From the realm of self-supervision I have experimented with self-training.

In more detail, the experiments I have performed are the following:
- Supervised
    - plain model
    - curriculum learning
    - contrastive learning
    - ensemble learning
    - curriculum + contrastive learning
    - ensemble + curriculum learning
    - ensemble + contrastive learning
    - ensemble + curriculum + contrastive learning

- Semi-supervised
    - self-training + plain model
    - self-training + curriculum learning
    - self-training + contrastive learning
    - self-training + ensemble learning
    - self-training + ensemble + curriculum learning
    - self-training + ensemble + contrastive learning
    - self-training + curriculum + contrastive learning
    - self-training + ensemble + curriculum + contrastive learning

## Models & Datasets

### Models

Throughout my experiments, I have worked with the following models:
- NER
    - [`bert-base-romanian-ner`](https://huggingface.co/dumitrescustefan/bert-base-romanian-ner)
    - [`bert-base-romanian-uncased-v1`](https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1)
    - [`bert-base-romanian-cased-v1`](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)
- Fake News Detection
    - [`bert-base-romanian-cased-v1`](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)
    - [`readerbench/RoBERT-base`](https://huggingface.co/readerbench/RoBERT-base)
    - [`readerbench/RoBERT-large`](https://huggingface.co/readerbench/RoBERT-large)

### Datasets and Data Preprocessing

For NER, I have used the [LegalNERo](https://huggingface.co/datasets/joelniklaus/legalnero) dataset (a subset of the Romanian MARCELL dataset) for providing 
labelled training examples and the [JuRO](https://github.com/craciuncg/GRAF) dataset for self-supervision. Since the JuRO dataset contains questions and answers 
from legal-domain college admission exams in Romania (the dataset was created for legal question answering), in order to obtain useful sentences for NER, I have 
concatenated each question with the longest correct answer. I have done this so as to ensure that the NER tagger has sufficiently many tokens to learn how to 
assign the labels from.

For fake news detection, I have used the [Ro-FakeNews](https://huggingface.co/datasets/upb-nlp/ro_fake_news) dataset, which contains both labelled and 
unlabelled training examples. This renders the dataset suitable for both supervised and semi-supervised training.

## Results

The results showed that, in terms of F1 scores, all the architectures I have tested are outperformed by the state-of-the-art by around 20%. The reason for this 
is that I had to stick with BERT as the base model, as the memory constraints of the clusters I ran these experiments on did not allow for training such 
configurations with any larger base model. This means that every architecture I have tested has under 1B parameters, unlike the LLM used for the state-of-the-
art, which have at least 7B parameters (Mistral).

## Reproducibility

For replicating the results of the experiments, clone the repository and run, from the root folder of the project, a command like the following: `python3 -m src.experiments.curriculum_learning legal_ner`. The command above trains and runs a single supervised model with curriculum learning for the NER task. To train 
and test the same architecture on the fake news detection task, just replace `legal_ner` with `fake_news`.

The hyperparameter values that rendered the highest F1 scores in my experiments are used as the defaults in the source code for each experiment. If you wish to 
try out other values, you can use command line arguments (added to the end of the example command). The available options are `--num_epochs`, `--num_workers`, `-
-lr`, `--batch_size`, `--model_name` and `--logging`. This last argument controls the outputs of the code in the [aim](https://github.com/aimhubio/aim) logging 
interface it is connected to.
