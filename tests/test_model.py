import os

import pytest
import torch
from hydra import compose, initialize

from src.model import IMDBTransformer
from tests import _PROJECT_ROOT


@pytest.mark.skipif(
    not os.path.exists(_PROJECT_ROOT + "/configs"), reason="Config files not found"
 )
#@pytest.mark.skipif(True)
def test_distil_model_output_shape():
    with initialize("../configs/", version_base=None):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "distilbert"
        model = IMDBTransformer(cfg)
        token_len = cfg["build_features"]["max_sequence_length"]
        x = torch.randint(0, 1000, (5, token_len))
        y = torch.randint(0, 1000, (5, token_len))
        (logits,) = model((x, y))

        assert logits.shape == torch.Size([5, 2])


@pytest.mark.skipif(
    not os.path.exists(_PROJECT_ROOT + "/configs"), reason="Config files not found"
)
# @pytest.mark.skipif(True)
def test_distil_model_is_default():
    with initialize("../configs/", version_base=None):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "non-existing-model"
        model = IMDBTransformer(cfg)

        assert "BertForSequenceClassification" in str(type(model.model))
