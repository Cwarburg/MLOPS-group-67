"""
train.py

This script makes the model training process:
1. Loads and preprocesses the IMDB dataset from data.py
2. Creates a classification model from model.py
3. Sets up a Hugging Face Trainer with Weights & Biases tracking
4. Trains the model and logs metrics/checkpoints

Usage:
    python train.py
"""

import wandb
from transformers import TrainingArguments, Trainer
from data import load_imdb_dataset
from model import create_model

def train_model(
    model_checkpoint="bert-base-uncased",
    output_dir="./models/bert-imdb",
    num_train_epochs=3,
    learning_rate=2e-5,
    train_batch_size=8,
    eval_batch_size=8,
    project_name="imdb-classification",
    run_name="bert-imdb-run1.0(toekn8)",
):
    """
    Trains a Transformer model on the IMDB dataset and logs metrics to Weights & Biases.

    Args:
        model_checkpoint (str): Hugging Face model name/path (e.g. "bert-base-uncased").
        output_dir (str): Where to store model checkpoints.
        num_train_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate.
        train_batch_size (int): Batch size for training.
        eval_batch_size (int): Batch size for evaluation.
        project_name (str): Weights & Biases project name.
        run_name (str): Weights & Biases run name.
    """

    # 1. Initialize Weights & Biases
    wandb.init(project=project_name, name=run_name)

    # 2. Load preprocessed IMDB data (tokenized + formatted for PyTorch)
    datasets = load_imdb_dataset(model_checkpoint=model_checkpoint)
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

    # 3. Create the model
    model = create_model(model_checkpoint=model_checkpoint, num_labels=2)

    # 4. Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoint at the end of each epoch
        logging_steps=100,              # Log training stats every 100 steps
        report_to=["wandb"],            # Enable Weights & Biases logging
        load_best_model_at_end=False,   # Set True if you want to load best model after training
        metric_for_best_model="accuracy",  # used if load_best_model_at_end=True
    )

    # 5. Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
        # If you have a compute_metrics function, you can pass it here:
        # compute_metrics=compute_metrics,
    )

    # 6. Train the model
    trainer.train()

    # 7. Evaluate the model
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    # 8. Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    # You can set different parameters here or pass them in from CLI if you wish.
    train_model()
