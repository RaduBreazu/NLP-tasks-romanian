from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from datasets import load_dataset

class MarcellDataset(Dataset):
    num_labels = 11 # the number of labels in the dataset
    
    def __init__(self, data : pd.DataFrame) -> None:
        data["text"] = data["text"].str.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
        to_int_labels = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-TIME": 7,
            "I-TIME": 8,
            "B-LEGAL": 9,
            "I-LEGAL": 10
        }
        data["label"] = data["label"].apply(np.vectorize(lambda x: to_int_labels.get(x, 0)))
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str | int]:
        row = self.data.iloc[idx, :]
        return {"text": row["text"], "label": row["label"]}
    
    @staticmethod
    def collator(batch : List[Dict[str, str | int]]) -> Dict[str, List[str] | List[int]]:
        return {"text": [item["text"] for item in batch], "label": [item["label"] for item in batch]}
    
    @staticmethod
    def to_string_label(label : int) -> str:
        to_str_labels = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-TIME",
            8: "I-TIME",
            9: "B-LEGAL",
            10: "I-LEGAL"
        }
        return to_str_labels.get(label, "O")

def join_words(df : pd.DataFrame) -> pd.DataFrame:
    df["text"] = [" ".join(word_list) for word_list in df["text"]]
    return df

def read_legalnero_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset("joelniklaus/legalnero")
    df = df.select_columns(["words", "ner"])
    df = df.rename_columns({"words": "text", "ner": "label"})
    df = df.map(join_words, batched = True)

    train_data = df["train"].to_pandas()
    validation_data = df["validation"].to_pandas()
    test_data = df["test"].to_pandas()

    return train_data, validation_data, test_data

def get_num_tokens(df : pd.DataFrame) -> Tuple[int, int]:
    total_tokens = sum(df["label"].apply(len))
    named_entity_tokens = sum(df["label"].apply(lambda arr: len(arr[arr != "O"])))
    return total_tokens, named_entity_tokens

def print_dataset_statistics() -> None:
    train_data, validation_data, test_data = read_legalnero_dataset()
    print("Dataset statistics:")
    [train_tokens, validation_tokens, test_tokens] = [get_num_tokens(data) for data in [train_data, validation_data, test_data]]
    print(f"Training data: {len(train_data)} samples, {train_tokens[0]} total tokens, {train_tokens[1]} named entity tokens")
    print(f"Validation data: {len(validation_data)} samples, {validation_tokens[0]} total tokens, {validation_tokens[1]} named entity tokens")
    print(f"Test data: {len(test_data)} samples, {test_tokens[0]} total tokens, {test_tokens[1]} named entity tokens")

def test_interaction_with_dataset() -> None:
    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")
    
    train_data, validation_data, test_data = read_legalnero_dataset()
    train_dataset = MarcellDataset(train_data)
    print(train_dataset[4])
    print(print_dict(MarcellDataset.collator(train_dataset[0:16])))

def main() -> None:
    print_statistics = True
    if print_statistics:
        print_dataset_statistics()
    else:
        test_interaction_with_dataset()

if __name__ == "__main__":
    main()