import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer

class IMDBTransformer(LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", 
                                                                        num_labels = 2, 
                                                                        num_labels = 2)
    def forward(self, batch):
        b_inputs_ids = batch
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
        b_labels = batch[1]
        (test_loss, logits) = self.model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        preds = torch.argmax(logits, dim=1)
        correct = (preds == b_labels).sum()
        accuracy = correct / len(b_labels)
    
        return {"loss": test_loss, "preds": preds, "labels": b_labels}