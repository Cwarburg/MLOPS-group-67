import os 
import pickle
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class IMDBReviews(Dataset):
    """
    Dataset class that loads the files created by make_dataset.py
    """

    def __init__(self, path: str, type : str = "train"):
        if type == "train":
            file = os.path.join(path, "train_tokenized.pkl")
        elif type == "test":
            file = os.path.join(path, "test_tokenized.pkl")
        elif type == "eval":
            file = os.path.join(path, "eval_tokenized.pkl")
        else:
            raise Exception(f"Unknown Dataset type : {type}")

        with open(file, 'rb') as f:
            print(f)
            data = pickle.load(f)

        self.reviews = torch.tensor(data['input_ids']) 
        self.masks = torch.tensor(data['attention_mask'])
        self.labels = torch.tensor(data['label'])


    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.reviews[idx],
            self.masks[idx],
            self.labels[idx],
        )

class IMDBReviewsModule(pl.LightningDataModule):
    
    def __init__(self, data_path : str, batch_size : int = 32):
        super().__init__()
        self.data_path = os.path.join(data_path, "processed")
        self.batch_size = batch_size
        self.cpu_cnt = os.cpu_count() or 2
    
    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_path):
            raise Exception("Data not prepared")
    
    def setup(self, stage : Optional[str] = None) -> None:
        self.trainset = IMDBReviews(self.data_path, "train")
        self.testset = IMDBReviews(self.data_path, "test")
        self.evalset = IMDBReviews(self.data_path, "eval")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size = self.batch_size, num_workers=self.cpu_cnt, persistent_workers=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset, batch_size = self.batch_size, num_workers=self.cpu_cnt, persistent_workers=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.evalset, batch_size=self.batch_size, num_workers=self.cpu_cnt, persistent_workers=True,
        )