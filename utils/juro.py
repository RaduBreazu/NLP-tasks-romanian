import os

from typing import Tuple, List, Dict

import pandas as pd

from torch.utils.data import Dataset

from .marcell import read_legalnero_dataset

class JuRODataset(Dataset):
    def __init__(self, data : pd.DataFrame) -> None:
        data["text"] = data["text"].str.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str | int]:
        row = self.data.iloc[idx, :]
        return {"idx": idx, "text": row["text"]}
    
    @staticmethod
    def collator(batch : List[Dict[str, str | int]]) -> Dict[str, List[str] | List[int]]:
        return {"idx": [item["idx"] for item in batch], "text": [item["text"] for item in batch]}


def get_all_filenames(type : str) -> List[str]:
    # determine the relative path (w.r.t to the project root, i.e. the `algorithms` folder)
    match type:
        case "train":
            file_path = "datasets/JuRO/train_set"
        case "validation":
            file_path = "datasets/JuRO/validation_set"
        case "test":
            file_path = "datasets/JuRO/test_set"
    
    files = []
    for dirpath, dirnames, filenames in os.walk(file_path):
        files.extend([os.path.join(dirpath, filename) for filename in filenames])

    return files

def df_from_row(row : Tuple[str, str, str, str]) -> pd.DataFrame:
    question = row[1][(row[1].find(".") + 3) :]
    first_part = question.strip(" :\t\n")
    
    index_a = row[2].find("A.")
    index_b = row[2].find("\nB.", index_a + 4) # see the comment below
    index_c = row[2].find("\nC.", index_b + 4)
    
    # when delimiting the answers, we need to strip the two whitespaces after "A.", "B." and "C."
    answer_a = row[2][(index_a + 3) : index_b] if index_b >= 0 else ""
    answer_b = row[2][(index_b + 3) : index_c] if index_b >= 0 else ""
    answer_c = row[2][(index_c + 3) :] if index_c >= 0 else ""
    match row[3][0]:
        case "A":
            first_correct = answer_a[1:] if len(answer_a) > 0 and answer_a[0] == " " else answer_a
        case "B":
            first_correct = answer_b[1:] if len(answer_b) > 0 and answer_b[0] == " " else answer_b
        case "C":
            first_correct = answer_c[1:] if len(answer_c) > 0 and answer_c[0] == " " else answer_c

    if first_correct == "":
        return pd.DataFrame({})
    
    # remove double spaces between sentence parts and replace punctuation with full stop (or add full stop, if necessary)
    sentence = (first_part + " " + first_correct).strip("\n")
    sentence = sentence + "." if sentence[-1].isalpha() else (sentence[:-1] + "." if sentence[-1] != "." else sentence)
    sentence = sentence[:-1] if sentence.endswith("..") else sentence
    
    return pd.DataFrame({'text': [sentence]})

def process_file(file : str) -> pd.DataFrame:
    df = pd.read_csv(file)
    dfs = [df_from_row(row) for row in df.itertuples(index = False, name = None)]
    new_df = pd.concat(dfs, ignore_index = True)
    return new_df

def get_dataset(type : str) -> pd.DataFrame:
    dfs = [process_file(file) for file in get_all_filenames(type)]
    return pd.concat(dfs, ignore_index = True)

def read_juro_dataset() -> pd.DataFrame:
    all_data = [get_dataset(type) for type in ["train", "validation", "test"]]
    unlabelled_data = pd.concat(all_data, ignore_index = True)
    return unlabelled_data

def read_legal_ner_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, validation_data, test_data = read_legalnero_dataset()
    unlabelled_data = read_juro_dataset()
    return train_data, validation_data, test_data, unlabelled_data

def main() -> None:
    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")

    _, _, _, unlabelled_data = read_legal_ner_dataset()
    unlabelled_dataset = JuRODataset(unlabelled_data)
    print(unlabelled_dataset[4])
    print(print_dict(JuRODataset.collator(unlabelled_dataset[0:16])))

if __name__ == "__main__":
    main()