from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import TrainingArguments
import torch
import pickle
import sys
import datasets
import os
from trl import DPOTrainer
import accelerate
import pandas as pd


if __name__ == '__main__':
    loss = sys.argv[0]
    max_length = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    path_to_dataset = 'data/train_dataset.pkl'
    with open(path_to_dataset, 'rb') as f:
        train_dataset = pickle.load(f)

    max_length = 128

    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", padding='max_length', truncation=True,
                                              max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    training_args = TrainingArguments(
        remove_unused_columns=False,
        learning_rate=0.005,
        evaluation_strategy='no',
        logging_steps=10,
        eval_steps=0,
        output_dir="./test",
        optim="rmsprop",
        warmup_steps=50,
        bf16=False,
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        max_prompt_length=1,
        max_length=256,
        loss_type=loss,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    dpo_trainer.save_model("models/hinge")


