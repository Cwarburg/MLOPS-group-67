# src/mlopsgroup67/model.py

"""
model.py

This module defines functions to create/configure a Transformer-based model
for sequence classification. All training logic should be done in train.py, 
and evaluation logic in evaluate.py.
"""

from transformers import AutoModelForSequenceClassification
models = ['tiny-bert',"bert-base-uncased"]
def create_model(model_checkpoint: str = 'tiny-bert', num_labels: int = 2):
    """
    Creates a sequence classification model from a pretrained checkpoint.

    Args:
        model_checkpoint (str): The name or path of a Hugging Face model checkpoint.
                               e.g. "bert-base-uncased" or "distilbert-base-uncased".
        num_labels (int): Number of output labels for the classification task.
                          For IMDB sentiment analysis, this is 2 (positive/negative).

    Returns:
        model (transformers.PreTrainedModel): An AutoModelForSequenceClassification instance
                                             ready to be trained.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )
    return model





