from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
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

    percentage_change = ((price_close - price_open) / price_open) * 100
    # Scale the percentage change to a range of 0 to 10
    scaled_value = max(-10, min(10, percentage_change))
    # Normalize the value to 0 to 10
    normalized_value = (scaled_value + 10) / 2
    label = int(normalized_value)
    if label != 5:
        print("Changed price!")

    print(f"appending text: {text}")
    data['text'].append(text)
    print(f"appending label: {label}")
    data['label'].append(label)

# reverse so the most recent come last
reversed_data = {k: v[::-1] for k, v in data.items()}

# Split data into train, validation, and test sets
train_data = {k: v[:int(len(v) * 0.8)] for k, v in reversed_data.items()}
val_data = {k: v[int(len(v) * 0.8):int(len(v) * 0.9)]
            for k, v in reversed_data.items()}
test_data = {k: v[int(len(v) * 0.9):] for k, v in reversed_data.items()}


# Create custom datasets
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


# Preprocess the dataset
tokenizer = BertTokenizerFast.from_pretrained(model)
config = BertConfig.from_pretrained(model, num_labels=11)

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained(
    model, config=config, ignore_mismatched_sizes=True)

# Tokenize each dataset separately
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

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

# Save the fine-tuned model
trainer.save_model("./results/checkpoint-1")
