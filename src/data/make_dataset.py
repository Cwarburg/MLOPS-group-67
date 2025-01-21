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


@hydra.main(config_path="../../configs", config_name="default_config.yaml")
def main(config : DictConfig) -> None :

    dataset_path = os.path.join(hydra.utils.get_original_cwd(), config.data.path)
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # TODO: get from config file instead of passed string 

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding = 'max_length', truncation = True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True) # Has features : text, label, input_ids, attention_mask
    
    train = os.path.join(dataset_path, "processed", "train_tokenized.pkl")
    test = os.path.join(dataset_path, "processed", "test_tokenized.pkl")
    eval = os.path.join(dataset_path, "processed", "eval_tokenized.pkl")

    with open(train, 'wb') as f:
        pickle.dump(tokenized_imdb['train'], f)

    with open(test, 'wb') as f:
        pickle.dump(tokenized_imdb['test'], f)

    with open(eval, 'wb') as f:
        pickle.dump(tokenized_imdb['unsupervised'], f)

# if __name__ == "__main__":
#     project_dir = Path(__file__).resolve().parents[2]
main()