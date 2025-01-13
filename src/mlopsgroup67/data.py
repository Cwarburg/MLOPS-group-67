# src/project_name/data.py

"""
data.py

This module handles loading and preprocessing of the IMDB dataset using Hugging Face Datasets.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def load_imdb_dataset(model_checkpoint: str = "bert-base-uncased", max_length: int = 8):
    """
    Loads the IMDB dataset from Hugging Face Datasets, tokenizes the texts, and returns
    the dataset in a PyTorch-ready format.

    Args:
        model_checkpoint (str): Name of the Hugging Face model checkpoint to use for tokenization.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        encoded_datasets (DatasetDict): A Hugging Face DatasetDict with 'train' and 'test' splits,
                                        containing tokenized input_ids, attention_mask, and labels.
    """
    # 1. Load raw IMDB dataset
    raw_datasets = load_dataset("imdb")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 3. Tokenizer function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    # 4. Apply tokenizer to the dataset
    encoded_datasets = raw_datasets.map(tokenize_function, batched=True)

    # 5. Set format to PyTorch
    encoded_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return encoded_datasets


if __name__ == "__main__":
    # Simple test of dataset loading
    ds = load_imdb_dataset()
    print(ds)
