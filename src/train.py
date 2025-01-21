import os 
from typing import Optional

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
# from dotenv import find_dotenv, load_dotenv
# from google.cloud import secretmanager
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data.dataset import IMDBReviewsModule
from src.model import IMDBTransformer

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config: DictConfig):
    
    data_module = IMDBReviewsModule(os.path.join(hydra.utils.get_original_cwd(), 
                                                 config.data_path), 
                                    batch_size=config.train.batch_size)
    data_module.setup()
    model = IMDBTransformer(config)

    gpus = 0
    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.device_count()} GPUs.")
        gpus = -1
    else:
        print("Training on CPU.")

    trainer = Trainer(
        max_epochs = config.train.epochs,
        gpus=gpus,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.test_dataloader(),
    )

if __name__=="__main__":

    project_dir = Path(__file__).resolve().parents[2]
    # load_dotenv(find_dotenv())

    main()