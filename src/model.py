import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import BertForSequenceClassification


class IMDBTransformer(LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", 
                                                                        num_labels = 2)
    
    def forward(self, batch):
        b_inputs_ids = batch[0]
        b_input_mask = batch[1]
        return self.model(b_inputs_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        (loss, _) = self.model(
            b_input_ids, 
            token_type_ids = None,
            attention_mask = b_input_mask,
            labels = b_labels
        )
        return loss
    
    def test_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        (test_loss, logits) = self.model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        preds = torch.argmax(logits, dim=1)
        correct = (preds == b_labels).sum()
        accuracy = correct / len(b_labels)
    
        return {"loss": test_loss, "accurace": accuracy, "preds": preds, "labels": b_labels}
    
    def setup(self, stage=None) -> None:
        pass

    # Define configure_optimizers function for defining which optimizer to use in training

    # Define save_jit to save deployable model to .pt file  