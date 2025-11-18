from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# ----------------------------
# Load Model
# ----------------------------
MODEL_NAME = "mrigaanksh/priority-classification-distilbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ----------------------------
# Request Body Schema
# ----------------------------
class InputText(BaseModel):
    text: str

# ----------------------------
# Prediction Route
# ----------------------------
@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # tokenize text
    inputs = tokenizer(text, return_tensors="pt")

    # model inference
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]

    confidence, label_id = torch.max(probs, dim=0)
    confidence = float(confidence * 100)
    label = f"LABEL_{label_id.item()}"

    return {
        "input": text,
        "label": label,
        "confidence": confidence
    }

# ----------------------------
# Root route
# ----------------------------
@app.get("/")
def home():
    return {"message": "Priority Classification BERT API is running!"}
