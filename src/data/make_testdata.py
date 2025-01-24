import logging
import os
import zipfile
from pathlib import Path

from hydra import initialize, compose
import pickle
import torch
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def load_test_dataset(config : DictConfig):

    # data_path = os.path.join(config.data.path, "processed")
    dataset_path = os.path.join(config.data.path)

    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # TODO: get from config file instead of passed string

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding = 'max_length', truncation = True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True) # Has features : text, label, input_ids, attention_mask

    tokenized_imdb.cleanup_cache_files()

    train = os.path.join(dataset_path, "processed", "train_tokenized.pkl")
    test = os.path.join(dataset_path, "processed", "test_tokenized.pkl")
    eval = os.path.join(dataset_path, "processed", "eval_tokenized.pkl")
    # tokenized_imdb.save_to_disk()

    with open(train, 'wb') as f:
        pickle.dump(tokenized_imdb['train'], f)

    with open(test, 'wb') as f:
        pickle.dump(tokenized_imdb['test'], f)

    with open(eval, 'wb') as f:
        pickle.dump(tokenized_imdb['unsupervised'], f)
