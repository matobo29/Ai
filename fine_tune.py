import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Load your dataset from Excel
def load_data(file_path):
    df = pd.read_excel(file_path)
    data = {
        "input": df["Question"].tolist(),
        "output": df["Answer"].tolist()
    }
    return Dataset.from_dict(data)

# Load dataset from Excel file
dataset = load_data("RSL_FAQs.xlsx")

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token to eos_token to avoid errors during tokenization
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(examples["input"], padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()  # Set labels as input_ids
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
