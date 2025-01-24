import glob
import logging
import os
from pathlib import Path
from time import time

import hydra
import numpy as np
import pandas as pd
import omegaconf
import torch.quantization
# from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from torch import nn
from transformers import AutoTokenizer

from data.dataset import IMDBReviewsModule
from model import IMDBTransformer


@hydra.main(config_path="../configs", config_name="default_config.yaml", version_base=None)
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Executing predict model script.")

    # Validate output folder
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), config.predict.model_output_dir
    )
    if not os.path.isdir(output_dir):
        raise Exception(
            'The "model_output_dir" path ({}) could not be found'.format(output_dir)
        )

    # Load local config in output directory
    output_config_path = os.path.join(output_dir, ".hydra", "config.yaml")
    output_config = omegaconf.OmegaConf.load(output_config_path)

    # Load model
    logger.info("Load model...")
    output_checkpoints_paths = os.path.join(
        output_dir, "lightning_logs", "version_4", "checkpoints", "*.ckpt"
    )
    output_checkpoint_latest_path = sorted(
        filter(os.path.isfile, glob.glob(output_checkpoints_paths))
    )[-1]
    model = IMDBTransformer.load_from_checkpoint(output_checkpoint_latest_path, config=output_config)
    # model

    # Load data module and use Validation data
    logger.info("Load data...")
    data_module = IMDBReviewsModule(
        os.path.join(hydra.utils.get_original_cwd(), config.data.path),
        batch_size=config.train.batch_size,
    )
    data_module.setup()
    data = data_module.evalset

    output_prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)

    if config.predict.quantization:
        logger.info("Initiating quantization...")
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        logger.info("Model converted for quantization.")

    logger.info("Predicting...")
    start_time = time()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    reviews = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in data.reviews[:20]]
    (y_pred,) = model([data.reviews[:20], data.masks[:20]], return_dict=False)
    # y_pred_np = y_pred.logits.detach().numpy()
    y_pred = torch.argmax(y_pred, dim=1)

    output_prediction_file = os.path.join(output_prediction_dir, "predictions.csv")
    # output_data = np.column_stack((reviews, y_pred.numpy()))

    df = pd.DataFrame({
        'Review': reviews,
        'Prediction': y_pred.numpy()
    })
    df.to_csv(output_prediction_file, index=False)
    # np.savetxt(output_prediction_file, output_data, delimiter=",", fmt="%s")

    logger.info(
        "Predictions are finished in {} seconds!\n Saved to {}".format(
            round(time() - start_time), output_prediction_file
        )
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()
