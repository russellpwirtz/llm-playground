import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class CryptoPredictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)

    def predict(self, input_sequence):
        # Create a pipeline with the tokenizer and model
        nlp = pipeline("text-classification",
                       tokenizer=self.tokenizer, model=self.model)

        # Use the pipeline to make predictions
        # text = "Replace this text with the one you want to classify"
        predictions = nlp(input_sequence)

        # Print the predictions
        print(predictions)
        inputs = self.tokenizer(
            input_sequence, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        label = torch.argmax(probabilities).item()
        return label


# model_path = 'results/checkpoint-6'
model_path = 'results/checkpoint-final'

predictor = CryptoPredictor(model_path)

# Example input sequence
input_sequence = "Symbol: ALB Open: 0.00168337 High: 0.00168337 Low: 0.00168337 Close: 0.00168337 Timestamp: 1681403760 Date: 2023-04-13T16:36:00"

# Make a prediction
predicted_label = predictor.predict(input_sequence)
print(f"Predicted label: {predicted_label}")
