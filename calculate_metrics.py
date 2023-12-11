import sys
from matplotlib import pyplot
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import os
import gc
from text_generation import generate_samples, evaluate_samples
from scipy.stats import entropy
from collections import defaultdict
from datetime import datetime


def generate_and_evaluate(model, tokenizer, bert_model, bert_tokenizer, size):
    samples = []
    reward = []
    device = model.device
    while size > 0:
        new_samples = generate_samples(model, tokenizer, size=100)
        new_reward = evaluate_samples(bert_model, bert_tokenizer, new_samples)
        samples += new_samples
        reward += new_reward
        size -= 100

    return samples, reward



def token_entropy(generations, tokenizer):
    stats = defaultdict(int)
    num_tokens = 0
    for example in generations:
        tokens = tokenizer.encode(example)
        for t in tokens:
            if t == tokenizer.pad_token_id:
                continue
            stats[t] += 1
            num_tokens += 1
    for k in stats.keys():
        stats[k] /= num_tokens
    return entropy(list(stats.values()))


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
    metrics = {}

    samples, reward = generate_and_evaluate(model, tokenizer,
                                            model_reward, tokenizer_reward,
                                            size=5000)

    metrics["model avg reward"] = float(torch.tensor(reward).mean())
    metrics["model std reward"] = float(torch.tensor(reward).std())
    diversity = token_entropy(samples, tokenizer_ft)
    metrics["model diversity"] = diversity

    del model
    gc.collect()
    torch.cuda.empty_cache()

    test_samples, test_reward = generate_and_evaluate(model_ft, tokenizer_ft,
                                                      bert_model, bert_tokenizer,
                                                      size=5000)

    metrics["model_ft avg reward"] = float(torch.tensor(test_reward).mean())
    metrics["model_ft std reward"] = float(torch.tensor(test_reward).std())
    diversity = token_entropy(test_samples, tokenizer_ft)
    metrics["model_ft diversity"] = diversity
    del model_ft
    gc.collect()
    torch.cuda.empty_cache()

    x = test_reward
    y = reward

    bins = np.linspace(-4, 5, 100)

    pyplot.hist(x, bins, alpha=0.5, label='fine-tuned GPT2')
    pyplot.hist(y, bins, alpha=0.5, label='GPT2')
    pyplot.legend(loc='upper right')

    now = datetime.now()
    time = now.strftime("%d %m %Y %H:%M:%S")
    image_path = f"images/{loss} {beta} {time}.png"
    res_path = f"results/{loss} {beta} {time}.txt"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    pyplot.savefig(image_path)
    pyplot.show()
    with open(res_path, 'w') as f:
        print(metrics, file=f)


if __name__ == '__main__':
    loss = sys.argv[1]
    beta = sys.argv[2]
    max_length = 128

    bert_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", padding='max_length', truncation=True,
                                                   max_length=max_length, padding_side="left")
    bert_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", padding='max_length', truncation=True,
                                              max_length=max_length, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    model_fine_tuned = AutoModelForCausalLM.from_pretrained(f"models/{loss} beta={beta}")
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

