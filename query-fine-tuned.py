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
input_sequence = "Price data for symbol: ALB on date: 2022-08-01 at timestamp: 1200 => Open: 100 High: 120 Low: 95 Close: 110"

# Make a prediction
predicted_label = predictor.predict(input_sequence)
print(f"Predicted label: {predicted_label}")


# 1. An input sequence with a clear increase in closing price compared to the opening price:
#    Input: `"Price data for symbol: ALB on date: 2022-08-01 at timestamp: 1200 => Open: 100 High: 120 Low: 95 Close: 110"`
# 2. An input sequence with a clear decrease in closing price compared to the opening price:
#    Input: `"Price data for symbol: ALB on date: 2022-08-02 at timestamp: 1200 => Open: 110 High: 115 Low: 90 Close: 95"`
# 3. An input sequence where the closing price is equal to the opening price:
#    Input: `"Price data for symbol: ALB on date: 2022-08-03 at timestamp: 1200 => Open: 100 High: 110 Low: 90 Close: 100"`

# # Sample input for prediction
# symbol = 'ALB'
# future_date = '2022-12-31'
# future_timestamp = '09:30:00'
# future_opening_price = 100.0

# future_text = f"Price data for symbol: {symbol} on date: {future_date} at timestamp: {future_timestamp} => Open: {future_opening_price}"

# inputs = tokenizer(future_text, return_tensors="pt", padding="max_length", truncation=True)

# # Generate prediction
# with torch.no_grad():
#     outputs = model(**inputs)
#     prediction = outputs[0].numpy()[0][0]

# print(f"The predicted percentage change for {symbol} on {future_date} is {prediction:.2f}")
