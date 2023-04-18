from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from data import parse_csv

# Sample input sequences
symbol = 'ALB'
input_sequences = parse_csv(symbol)
model = 'flowfree/bert-finetuned-cryptos'

# Convert input sequences to a dictionary of lists
data = {'text': [], 'label': []}
for seq in input_sequences:
    price_open = seq['ohlc'][0]
    price_high = seq['ohlc'][1]
    price_low = seq['ohlc'][2]
    price_close = seq['ohlc'][3]
    text = f"Price data for symbol: {seq['symbol']} on date: {seq['date']} at timestamp: {seq['timestamp']} => Open: {price_open} High: {price_high} Low: {price_low} Close: {price_close}"
    label = 1 if price_close > price_open else 0
    print(f"appending text: {text}")
    data['text'].append(text)
    print(f"appending label: {label}")
    data['label'].append(label)

# Split data into train, validation, and test sets
train_data = {k: v[:int(len(v) * 0.8)] for k, v in data.items()}
val_data = {k: v[int(len(v) * 0.8):int(len(v) * 0.9)] for k, v in data.items()}
test_data = {k: v[int(len(v) * 0.9):] for k, v in data.items()}

# Create custom datasets
train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Replace the following path with the path to your fine-tuned model
model_checkpoint = "results/checkpoint-6"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Tokenize the dataset


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(
    tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./results/checkpoint-final")
