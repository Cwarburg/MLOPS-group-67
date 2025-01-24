import os
from typing import Optional

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # <-- Import the Wandb logger
from omegaconf import DictConfig

from data.dataset import IMDBReviewsModule
from model import IMDBTransformer

@hydra.main(config_path="../configs", config_name="default_config.yaml", version_base=None)
def main(config: DictConfig):

    # Initialize wandb logger. Customize "project" or "name" as you wish.
    wandb_logger = WandbLogger(
        project="my-awesome-project",    # e.g. "IMDB-Sentiment"
        name="experiment-with-transformer"
    )
    # Optionally, store your Hydra config in wandb's config
    # (so you can see hyperparameters in the wandb UI)
    wandb_logger.experiment.config.update(dict(config))

    # Prepare data module
    data_module = IMDBReviewsModule(
        os.path.join(hydra.utils.get_original_cwd(), config.data.path),
        batch_size=config.train.batch_size
    )
    data_module.setup()
    
    model = IMDBTransformer(config)

    # Check GPU availability
    gpus = 0
    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.device_count()} GPUs.")
        gpus = -1  # -1 means "use all available GPUs" in PyTorch Lightning
    else:
        print("Training on CPU.")

    # Create the trainer with the wandb logger
    trainer = Trainer(
        max_epochs=config.train.epochs,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        logger=wandb_logger   # <-- Pass in the wandb logger
        #gpus=gpus             # or accelerator="gpu", devices=-1 in Lightning >= 1.7
    )

    # 6. Fit the model
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.test_dataloader(),
    )

    # 7. Save model
    model.save_jit()

    # 8. (Optional) Explicitly finish wandb run
    # wandb.finish()  # Not strictly necessary; Lightning will close it at script exit.

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    main()
