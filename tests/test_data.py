import pytest
from datasets import DatasetDict, Dataset
from mlopsgroup67.data.make_dataset import load_imdb_dataset


def test_load_imdb_dataset():
    """Test the load_imdb_dataset function."""
    # Call the function with default arguments
    dataset = load_imdb_dataset()

    # Ensure the returned object is a DatasetDict
    assert isinstance(dataset, DatasetDict), "load_imdb_dataset should return a DatasetDict."

    # Check if the 'train' and 'test' splits exist
    assert "train" in dataset, "The dataset should contain a 'train' split."
    assert "test" in dataset, "The dataset should contain a 'test' split."

    # Check if the splits contain data
    assert len(dataset["train"]) > 0, "The 'train' split should not be empty."
    assert len(dataset["test"]) > 0, "The 'test' split should not be empty."

    # Check the format and columns of the dataset
    train_sample = dataset["train"][0]
    assert "input_ids" in train_sample, "Each example should have 'input_ids'."
    assert "attention_mask" in train_sample, "Each example should have 'attention_mask'."
    assert "label" in train_sample, "Each example should have 'label'."

    # Ensure the splits are instances of Hugging Face's Dataset
    assert isinstance(dataset["train"], Dataset), "The 'train' split should be a Hugging Face Dataset."
    assert isinstance(dataset["test"], Dataset), "The 'test' split should be a Hugging Face Dataset."
