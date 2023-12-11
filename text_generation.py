from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import pickle
import os


def generate_samples(model, tokenizer, prefix=None, size=1,):
    """
    Simple function for text generation:
    Input:
      model: model for text generation
      tokenizer: tokenizer of model
      prefix: first tokens
      size: number of samples to generate
    Output:
      text: list of generated texts
    """
    device = model.device
    if prefix:
        tokenized_prefix = tokenizer.encode(prefix, return_tensors='pt').to(device)
    else:
        tokenized_prefix = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    encoded_sample = model.generate(tokenized_prefix,
                                    do_sample=True,
                                    max_length=64,
                                    top_k=50,
                                    top_p=0.85,
                                    num_return_sequences=size,
                                    pad_token_id=tokenizer.eos_token_id)

    text = tokenizer.batch_decode(encoded_sample, skip_special_tokens=True)
    return text


def evaluate_samples(model, tokenizer, samples):
    """
    Calculate reward score of samples depending on model
    Input:
      model: classification model
      tokenizer: model tokenizer
      sample: list of text to evaluate
    Output:
      reward: torch.tensor, calculated logits of classification model
    """
    device = model.device
    enc = tokenizer.batch_encode_plus(samples, padding=True)
    reward = model(torch.tensor(enc['input_ids']).to(device),
                   torch.tensor(enc['attention_mask']).to(device)).logits[:, 1].tolist()
    #1 -- positive
    return reward


def generate_dataset(model, tokenizer, model_eval, tokenizer_eval, size=512):
    """
    :param model: model to generate data
    :param tokenizer:  tokenizer for generative model
    :param model_eval:  reward model
    :param tokenizer_eval:  tokenizer for reward model
    :param size:  size of dataset
    :return: List of tuples: (sample, reward)
    """
    evaluated_texts = []
    while size > 0:
        torch.cuda.empty_cache()
        samples_list = generate_samples(model, tokenizer, size=min(250, size))
        rewards = evaluate_samples(model_eval, tokenizer_eval, samples_list)
        evaluated_texts[:0] = [(text, reward) for text, reward in zip(samples_list, rewards)]
        size -= 250
    return evaluated_texts


if __name__ == '__main__':
    # generate texts and save it
    bert_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", padding='max_length', padding_side='left')
    bert_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", padding='max_length', padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    bert_model = bert_model.to(device)
    evaluated_texts = generate_dataset(model, tokenizer, bert_model, bert_tokenizer, size=10000)
    torch.cuda.empty_cache()
    filename = 'data/evaluated_texts.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(evaluated_texts, f)
