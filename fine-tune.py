from torch.nn import MSELoss
from datasets import Dataset
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from data import parse_csv

# Sample input sequences
symbol = 'ALB'
input_sequences = parse_csv(symbol)
model = 'flowfree/bert-finetuned-cryptos'

large_threshold = 0.05

# Convert input sequences to a dictionary of lists
data = {'text': [], 'label': []}
for seq in input_sequences:
    price_open = seq['ohlc'][0]
    price_high = seq['ohlc'][1]
    price_low = seq['ohlc'][2]
    price_close = seq['ohlc'][3]
    text = f"Price data for symbol: {seq['symbol']} on date: {seq['date']} at timestamp: {seq['timestamp']} => Open: {price_open} High: {price_high} Low: {price_low} Close: {price_close}"

#     label = 1 if price_close > price_open else 0
    percentage_change = (price_close - price_open) / price_open
    if percentage_change == 0:
        label = 0
    elif percentage_change > 0:
        label = min(percentage_change / large_threshold, 1)
    else:
        label = max(percentage_change / large_threshold, -1)

    data['text'].append(text)
    data['label'].append(label)

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
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Preprocess the dataset


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


class BertForSequenceRegression(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels),
                            labels.view(-1, self.config.num_labels))
            if return_dict:
                return_dict = {
                    "hidden_states": outputs.hidden_states,
                    "attentions": outputs.attentions,
                }
                return (loss,) + (BaseModelOutputWithPoolingAndCrossAttentions(**return_dict),)
            else:
                return (loss,) + tuple(outputs)

        return logits


tokenizer = BertTokenizerFast.from_pretrained(model)

config = BertConfig.from_pretrained(model, num_labels=1)
model = BertForSequenceRegression.from_pretrained(
    model, config=config, ignore_mismatched_sizes=True
)

# Initialize the classifier layer with random weights
model.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

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

test_results = trainer.evaluate(test_dataset)
print(test_results)

# Save the fine-tuned model
trainer.save_model("./results/checkpoint-1")


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
