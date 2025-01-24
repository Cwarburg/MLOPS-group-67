from fastapi import FastAPI
from typing import Dict
import torch

from transformers import BertForSequenceClassification, AutoTokenizer
import io

from data.dataset import IMDBReviewsModule
from model import IMDBTransformer

app = FastAPI()

@app.get("/")
def root():
    response = {"message": HTTPStatus.OK.phrase,
                "status_code": HTTPStatus.OK}
    return response

model = torch.jit.load("../deployable_model.pt")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.eval()

def encode(text):
    indices = tokenizer.encode_plus(
        text,
        max_length=64,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
    )

    return indices["input_ids"], indices["attention_mask"]

@app.post("/predict")
def model_predict(text: str):
    try:
        review, mask = encode(text)
        review = torch.IntTensor(review)
        mask = torch.LongTensor(mask)
        review.unsqueeze_(0)
        mask.unsqueeze_(0)
        (pred,) = model(review, mask)
        y_pred =  torch.argmax(pred, 1).item()

        return {"predicted_class": y_pred}
    except Exception as e:
        return {"error": str(e)}
