from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import torch
import os

model_path = "/content/trained_model"  # Replace with your model path or Hugging Face model name
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

model = model.float()

for param in model.parameters():
    param.requires_grad = True

data_1 = pd.read_csv('/content/python_data.csv', encoding='cp1252').rename(columns={"prompt": "instruction", "completion": "output"})
data_2 = pd.read_csv('/content/python_data_1.csv', encoding='cp1252').rename(columns={"Question": "instruction", "Answer": "output"})

combined_data = pd.concat([data_1, data_2], ignore_index=True)
combined_data['text'] = combined_data.apply(lambda row: f"You are an AI assistant... User: {row['instruction']} Assistant: {row['output']}", axis=1)

dataset = Dataset.from_pandas(combined_data[['text']])
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  # Reduced max_length to 128

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
validation_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",  # Enable evaluation at each epoch
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    remove_unused_columns=False,
    fp16=False,
    bf16=False,
    no_cuda=True  # Use CPU only
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,  # Use validation dataset
    tokenizer=tokenizer,
    data_collator=data_collator
)



trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
