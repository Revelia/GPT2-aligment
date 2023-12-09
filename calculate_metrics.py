import sys

from matplotlib import pyplot
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import TrainingArguments
import torch
import pickle
import datasets
import os
import gc
from trl import DPOTrainer
import accelerate
import pandas as pd
from text_generation import generate_samples, evaluate_samples


def calculate_metrics(model,
                      tokenizer,
                      model_ft,
                      tokenizer_ft,
                      model_reward,
                      tokenizer_reward,
                      image_name='result'):
    """
    :param model:
    :param tokenizer:
    :param model_ft:
    :param tokenizer_ft:
    :param model_reward:
    :param tokenizer_reward:
    :param image_name:
    :return:
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    samples = generate_samples(model, tokenizer, device, size=100)
    reward = evaluate_samples(model_reward, tokenizer_reward, samples, device)
    torch.tensor(reward).mean()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    test_samples = generate_samples(model_ft, tokenizer_ft, device, size=100)
    test_reward = evaluate_samples(model_reward, tokenizer_reward, test_samples, device)
    print(f"avg reward:{torch.tensor(test_reward).mean()}")

    del model_ft
    gc.collect()
    torch.cuda.empty_cache()

    x = test_reward
    y = reward

    bins = np.linspace(-4, 5, 100)

    pyplot.hist(x, bins, alpha=0.5, label='fine-tuned GPT2')
    pyplot.hist(y, bins, alpha=0.5, label='GPT2')
    pyplot.legend(loc='upper right')
    pyplot.show()
    image_path = f"images/{loss}"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    pyplot.savefig(image_path)

if __name__ == '__main__':
    loss = sys.argv[1]

    max_length = 128

    bert_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", padding='max_length', truncation=True,
                                                   max_length=max_length)
    bert_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", padding='max_length', truncation=True,
                                              max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    model_fine_tuned = AutoModelForCausalLM.from_pretrained(f"models/{loss}")
    model_fine_tuned.cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    bert_model = bert_model.to(device)

    calculate_metrics(model,
                      tokenizer,
                      model_fine_tuned,
                      tokenizer,
                      bert_model,
                      bert_tokenizer)

