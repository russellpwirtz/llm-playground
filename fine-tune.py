from datasets import Dataset
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

# Sample input sequences
input_sequences = []
symbol = 'ALB'
model = ''

# Convert input sequences to a dictionary of lists
data = {'text': [], 'label': []}
for seq in input_sequences:
    price_open, price_high, price_low, price_close, timestamp, date = seq
    text = f"Symbol: {symbol} Open: {price_open} High: {price_high} Low: {price_low} Close: {price_close} Timestamp: {timestamp} Date: {date}"
    label = 1 if price_close > price_open else 0
    data['text'].append(text)
    data['label'].append(label)

# Split data into train, validation, and test sets
train_data = {k: v[:int(len(v) * 0.8)] for k, v in data.items()}
val_data = {k: v[int(len(v) * 0.8):int(len(v) * 0.9)] for k, v in data.items()}
test_data = {k: v[int(len(v) * 0.9):] for k, v in data.items()}

# Create custom datasets
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Preprocess the dataset


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


tokenizer = BertTokenizerFast.from_pretrained(model)

# Tokenize each dataset separately
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the pre-trained model
config = BertConfig.from_pretrained(
    model, num_labels=3)
model = BertForSequenceClassification.from_pretrained(model, config=config)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
