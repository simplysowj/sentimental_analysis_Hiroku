import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import argparse

def load_and_prepare_data(file_path, test_size=0.2):
    """Load and split the dataset"""
    df = pd.read_csv(file_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

def tokenize_data(dataset, tokenizer):
    """Tokenize the dataset"""
    return dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
        batched=True
    )

def setup_lora_model(model_name, num_labels):
    """Set up the base model and apply LoRA"""
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train_model(model, tokenized_train, tokenized_val, tokenizer, output_dir):
    """Train the model with LoRA"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file with 'text' and 'label' columns")
    parser.add_argument("--output_dir", type=str, default="lora_bert_classifier", help="Output directory for model")
    args = parser.parse_args()

    # Constants
    MODEL_NAME = "bert-base-uncased"

    # Load and prepare data
    train_dataset, val_dataset = load_and_prepare_data(args.data)
    num_labels = len(pd.read_csv(args.data)['label'].unique())

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    tokenized_train = tokenize_data(train_dataset, tokenizer)
    tokenized_val = tokenize_data(val_dataset, tokenizer)

    # Set up LoRA model
    model = setup_lora_model(MODEL_NAME, num_labels)

    # Train the model
    trainer = train_model(model, tokenized_train, tokenized_val, tokenizer, args.output_dir)

    # Evaluate and save
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
