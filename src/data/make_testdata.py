import logging
import os
import zipfile
from pathlib import Path

from hydra import initialize, compose
import pickle
import torch
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig


def load_test_dataset(config : DictConfig):

    data_path = os.path.join(config.data.path, "processed")

    train = os.path.join(data_path, "train_test.pkl")
    test = os.path.join(data_path, "test_test.pkl")
    eval = os.path.join(data_path, "eval_test.pkl")

    with open(train, 'wb') as f:
        pickle.dump(torch.zeros(25000), f)

    with open(test, 'wb') as f:
        pickle.dump(torch.zeros(25000), f)

    with open(eval, 'wb') as f:
        pickle.dump(torch.zeros(50000), f)
