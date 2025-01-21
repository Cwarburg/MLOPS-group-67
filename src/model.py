import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from transformers import BertForSequenceClassification


class IMDBTransformer(LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(self.config.model["pretrained-model"], 
                                                                   torchscript = True,
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
            labels = b_labels,
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
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[object]]:
        if self.config.train["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr = float(self.config.train["lr"]),
                eps = float(self.config.train["eps"]),
                betas = (0.9, 0.999)
            )
        else: 
            raise ValueError("Unknown Optim")
        
        if self.config.train["scheduler"]["name"] == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config.train["scheduler"]["gamma"]
            )
        else:
            raise ValueError("Unknown Scheduler")
        return [optimizer], [scheduler]

    # Define save_jit to save deployable model to .pt file
    def save_jit(self, file: str = "deployable_model.pt") -> None:
        token_len = self.config["build_features"]["max_sequence_length"]
        tokens_tensor = torch.ones(1, token_len).long()
        mask_tensor = torch.ones(1, token_len).float()
        script_model = torch.jit.trace(self.model, [tokens_tensor, mask_tensor])
        script_model.save(file)