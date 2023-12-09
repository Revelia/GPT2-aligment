from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import pickle
import datasets
import pandas as pd
import random
import os


def build_pairs(data, size):
    dataset_dict = {"prompt": [], "chosen": [], "rejected": [],}

    for _ in range(size):
        first, second = random.sample(data, 2)
        # if reward of first sample is less than reward of second
        if first[1] < second[1]:
            first, second = second, first
        # we generate sentences without prefix
        dataset_dict["prompt"].append("")
        dataset_dict["chosen"].append(first[0])
        dataset_dict["rejected"].append(second[0])
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset_dict))

    return dataset


if __name__ == '__main__':
    path_to_raw = 'data/evaluated_texts.pkl'
    path_to_processed = 'data/train_dataset.pkl'
    with open(path_to_raw, 'rb') as f:
        evaluated_texts = pickle.load(f)
    train_dataset = build_pairs(evaluated_texts, 2000)

    with open(path_to_processed, 'wb') as f:
        pickle.dump(train_dataset, f)
