import logging
import os
import zipfile
from pathlib import Path

import hydra
import pickle
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer



def main() -> None :

    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # TODO: get from config file instead of passed string 

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding = 'max_length', truncation = True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True) # Has features : text, label, input_ids, attention_mask
    
    train = '../../../data/processed/train_tokenized.pkl'
    test = '../../../data/processed/test_tokenized.pkl'

    with open(train, 'wb') as f:
        pickle.dump(tokenized_imdb['train'], f)

    with open(test, 'wb') as f:
        pickle.dump(tokenized_imdb['test'], f)

main()