from typing import Tuple, List, Dict

import pandas as pd

from torch.utils.data import Dataset

from datasets import load_dataset

class FakeNewsDataset(Dataset):
    num_labels = 10 # the number of labels in the dataset
    
    def __init__(self, data : pd.DataFrame) -> None:
        data["headline"] = data["headline"].str.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
        data["body"] = data["body"].str.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
        try:
            to_int_labels = {
                'Știre reală': 0,
                'Narațiuni legate de vaccinare': 1,
                'Acuzații de autoritarism și distopie': 2,
                'Narațiuni legate de sănătate': 3,
                'Criticarea restricțiilor': 4,
                'Teorii ale conspirației': 5,
                'Geo-politică': 6,
                'Satiră': 7,
                'Minimizarea impactului COVID-19': 8,
                'Criticarea actorilor naționali și internaționali': 9
            }
            data["label"] = data["label"].map(to_int_labels)
        except KeyError:
            pass
        finally:
            self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx : int) -> Dict[str, str | int]:
        row = self.data.iloc[idx, :]
        output = {"headline": row["headline"], "body": row["body"]}
        try:
            label = row["label"]
            output["label"] = label
        except KeyError:
            pass
        finally:
            return output
        
    @staticmethod
    def collator(batch : List[Dict[str, str | int]]) -> Dict[str, List[str] | List[int]]:
        return {
            "text": [item["headline"] + "\n" + item["body"] for item in batch],
            "label": [item["label"] for item in batch]
        }
    
    @staticmethod
    def to_string_label(label : int) -> str:
        to_str_labels = {
            0: 'Știre reală',
            1: 'Narațiuni legate de vaccinare',
            2: 'Acuzații de autoritarism și distopie',
            3: 'Narațiuni legate de sănătate',
            4: 'Criticarea restricțiilor',
            5: 'Teorii ale conspirației',
            6: 'Geo-politică',
            7: 'Satiră',
            8: 'Minimizarea impactului COVID-19',
            9: 'Criticarea actorilor naționali și internaționali'
        }
        return to_str_labels.get(label, "Știre reală")


def read_fake_news_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled_data = load_dataset("upb-nlp/ro_fake_news", "default")
    unlabelled_data = load_dataset("upb-nlp/ro_fake_news", "unlabeled", split = "train").to_pandas()

    labelled_data = labelled_data.select_columns(['supernarrative', 'headline', 'body'])
    labelled_data = labelled_data.rename_columns({'supernarrative': 'label'})
    unlabelled_data = unlabelled_data.loc[:, ['headline', 'body']]

    labelled_train_data = labelled_data["train"].to_pandas()
    validation_data = labelled_data["validation"].to_pandas()
    labelled_test_data = labelled_data["test"].to_pandas()

    labelled_data = labelled_train_data.copy()
    test_data = labelled_test_data.copy()

    return labelled_data, validation_data, test_data, unlabelled_data

def main() -> None:
    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")
    
    train_data, validation_data, test_data, unlabelled_data = read_fake_news_dataset()
    train_dataset = FakeNewsDataset(train_data)
    print(train_dataset[4])
    print(type(train_dataset[0:16]["body"]))
    print(print_dict(FakeNewsDataset.collator(train_data[0:16])))

if __name__ == "__main__":
    main()