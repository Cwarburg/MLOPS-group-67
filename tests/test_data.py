import os
import pytest

from hydra import compose, initialize

from src.data.dataset import IMDBReviewsModule
from tests import _PATH_DATA, _PROJECT_ROOT
from src.data.make_testdata import *

@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA)
    or not os.path.exists(_PROJECT_ROOT + "/configs"),
    reason="Data and config files not found",
)
def test_load_imdb_dataset():
    """Test the load_imdb_dataset function."""
    # Call the function with default arguments
    with initialize("../configs/", version_base=None):
        
        config = compose(config_name="default_config.yaml")
        load_test_dataset(config)

        data_module = IMDBReviewsModule(os.path.join(_PATH_DATA), batch_size=config.train.batch_size)
        data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        train_set_len = len(train_loader.dataset)
        val_set_len = len(val_loader.dataset)
        test_set_len = len(test_loader.dataset)

        assert train_set_len + val_set_len + test_set_len == 100000