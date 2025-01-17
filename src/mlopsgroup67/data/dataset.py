import os 
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class IMDBReviews(Dataset):
    def __init__(self, path: str, type : str = "train"):
        if type == "train":
            file = os.path.join(path, "train_tokenized.pkl")
        elif type == "test":
            file = os.path.join(path, "test_tokenized.pkl")
        else:
            raise Exception(f"Unknown Dataset type : {type}")

        file = pickle.load(file)


class IMDBReviewsModule(pl.LightningDataModule):
    
