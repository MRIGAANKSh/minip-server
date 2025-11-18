from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI()

MODEL_ID = "mrigaanksh/priority-classification-distilbert"

# Load HuggingFace model + tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Map HuggingFace labels
priority_labels = ["low", "medium", "high"]

class RequestBody(BaseModel):
    text: str

@app.post("/api/predict")
def predict_priority(body: RequestBody):
    text = body.text

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Get top class
    idx = int(np.argmax(probs))
    priority = priority_labels[idx]
    confidence = float(probs[idx])

    return {
        "priority": priority,
        "confidence": confidence
    }
