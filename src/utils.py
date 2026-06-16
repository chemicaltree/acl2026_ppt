import os
import pickle
import random
from collections import defaultdict

import datasets
import fire
import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm, trange
from transformers import AutoTokenizer


def load_avg_activations(model, avg_activation_path, device):
    """
    Loads average activations from a pickle file and sets them on the model's modules.
    Assumes each module has a set_avg_activation method.
    """
    with open(avg_activation_path, "rb") as f:
        avg_activations = pickle.load(f)
    for name, module in model.named_modules():
        if name in avg_activations and hasattr(module, "set_avg_activation"):
            module.set_avg_activation(torch.tensor(avg_activations[name]).to(device))


def load_text_files(file_dir):
    texts = []
    # Read each file in the directory
    for filename in tqdm(os.listdir(file_dir)):
        if filename.endswith(".txt"):  # Adjust file extension if needed
            file_path = os.path.join(file_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return datasets.Dataset.from_dict({"text": texts})


def cache_data_txt(
    file_dir: str, out_dir: str, tokenizer_name: str = "EleutherAI/pythia-1b"
):
    # load text files into huggingface dataset
    dataset = load_text_files(file_dir)
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=1024
        ),
        batched=True,
    )

    dataset.save_to_disk(out_dir)


def cache_data(
    dataset_name: str = None,
    out_dir: str = None,
    tokenizer_name: str = "EleutherAI/pythia-1b",
    data_files: str = None,
):
    """
    Cache and tokenize dataset.

    Args:
        dataset_name: Single dataset name/path (for backward compatibility)
        out_dir: Output directory for cached data
        tokenizer_name: Name of tokenizer to use
        data_files: Comma-separated list of file paths (e.g., "file1.json,file2.json")
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # Handle the data_files parameter (multiple files)
    if data_files is not None:
        # Split comma-separated string into list
        files_list = [f.strip() for f in data_files.split(",")]
        dataset = load_dataset("json", data_files=files_list, split="train")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=1024,
            ),
        ).remove_columns(["text"])
    elif dataset_name is not None:
        # Original logic for single dataset_name
        if "verbatim" in dataset_name:
            dataset = load_dataset(
                "text",
                data_files=[
                    f"./data/categorized_lists_sce{i}_repeat.txt" for i in range(1, 6)
                ],
            )
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    max_length=1024,
                ),
            ).remove_columns(["text"])
        elif dataset_name == "blimp":
            dataset = datasets.load_dataset("WillHeld/blimp", split="train")

            def tokenize_examples(examples):
                good = tokenizer(
                    examples["sentence_good"],
                    truncation=True,
                    max_length=128,
                )
                bad = tokenizer(
                    examples["sentence_bad"],
                    truncation=True,
                    max_length=128,
                )
                return {
                    "good_input_ids": good["input_ids"],
                    "good_attention_mask": good["attention_mask"],
                    "bad_input_ids": bad["input_ids"],
                    "bad_attention_mask": bad["attention_mask"],
                }

            cols = dataset.column_names
            dataset = dataset.map(tokenize_examples, batched=True).remove_columns(cols)
        elif dataset_name == "wikitext":
            dataset = load_dataset(
                "Salesforce/wikitext", name="wikitext-2-v1", split="train"
            )
            dataset = dataset.map(
                lambda x: tokenizer(x["text"], truncation=True, max_length=1024),
                batched=True,
            )
        else:
            dataset = load_dataset("text", data_files=dataset_name, split="train")
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",                    
                    max_length=1024,
                ),
            ).remove_columns(["text"])
    else:
        raise ValueError("Either dataset_name or data_files must be provided")

    dataset.save_to_disk(out_dir)


def train_ngram_models(dataset):
    """
    Trains unigram, bigram, and trigram language models on a tokenized dataset.
    Modifies unigram model to exclude <s> from being generated after the start.

    Args:
      dataset: A tokenized Hugging Face dataset.

    Returns:
      A tuple of three dictionaries representing the unigram, bigram, and trigram models.
    """

    unigram_model = defaultdict(int)
    bigram_model = defaultdict(lambda: defaultdict(int))
    trigram_model = defaultdict(lambda: defaultdict(int))

    for example in tqdm(dataset):
        tokens = example["input_ids"]
        tokens = ["<s>"] + tokens  # Add start token

        for i in range(len(tokens)):
            unigram_model[tuple(tokens[i : i + 1])] += 1  # Unigram
            if i < len(tokens) - 1:
                bigram_model[tuple(tokens[i : i + 1])][tokens[i + 1]] += 1  # Bigram
            if i < len(tokens) - 2:
                trigram_model[tuple(tokens[i : i + 2])][tokens[i + 2]] += 1  # Trigram

    # --- Modification to prevent <s> generation after start ---
    unigram_model_no_s = unigram_model.copy()
    del unigram_model_no_s[("<s>",)]  # Remove <s> from unigram model

    # Normalize counts to probabilities

    # Unigram Normalization (for generation without <s>)
    total_unigram_count = sum(unigram_model_no_s.values())
    for unigram in unigram_model_no_s:
        unigram_model_no_s[unigram] /= total_unigram_count

    # Bigram and Trigram Normalization
    for model in [bigram_model, trigram_model]:
        for prev_words in model:
            total_count = sum(model[prev_words].values())
            for word in model[prev_words]:
                model[prev_words][word] /= total_count

    # Return the modified unigram model without <s>
    return unigram_model_no_s, bigram_model, trigram_model


def generate_text(
    unigram_model,
    bigram_model,
    trigram_model,
    use_trigram=False,
    use_bigram=False,
    length=20,
):
    """
    Generates text using a trigram model with backoff to bigram and unigram models.
    Uses sampling instead of argmax.

    Args:
      unigram_model: The trained unigram model.
      bigram_model: The trained bigram model.
      trigram_model: The trained trigram model.
      use_trigram: Whether to use the trigram model if available.
      use_bigram: Whether to use the bigram model if available.
      length: The desired length of the generated text (in tokens).

    Returns:
      A list containing the generated tokens (excluding the start token).
    """

    text = ["<s>"]

    for _ in range(length):
        if len(text) >= 2 and tuple(text[-2:]) in trigram_model and use_trigram:
            # Trigram sampling
            candidates = list(trigram_model[tuple(text[-2:])].keys())
            probabilities = list(trigram_model[tuple(text[-2:])].values())
            next_word = random.choices(candidates, probabilities)[0]
        elif len(text) >= 1 and tuple(text[-1:]) in bigram_model and use_bigram:
            # Bigram sampling
            candidates = list(bigram_model[tuple(text[-1:])].keys())
            probabilities = list(bigram_model[tuple(text[-1:])].values())
            next_word = random.choices(candidates, probabilities)[0]
        else:
            # Unigram sampling
            candidates = list(unigram_model.keys())
            probabilities = list(unigram_model.values())
            next_word = random.choices(candidates, probabilities)[0][0]

        text.append(next_word)

    return text[1:]


def make_dummy_tasks(data_dir, out_dir, length=2048):
    dataset = load_from_disk(data_dir)
    unigram_model, bigram_model, trigram_model = train_ngram_models(dataset)

    # Generate text for each row and store in a list
    new_unigram = []
    new_bigram = []
    new_trigram = []

    for i in trange(len(dataset)):
        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=True,
            use_bigram=True,
            length=2048,
        )
        new_trigram.append({"input_ids": text})

        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=False,
            use_bigram=True,
            length=2048,
        )
        new_bigram.append({"input_ids": text})

        text = generate_text(
            unigram_model,
            bigram_model,
            trigram_model,
            use_trigram=False,
            use_bigram=False,
            length=2048,
        )
        new_unigram.append({"input_ids": text})

    unigram_dataset = Dataset.from_list(new_unigram)
    bigram_dataset = Dataset.from_list(new_bigram)
    trigram_dataset = Dataset.from_list(new_trigram)

    unigram_dataset.save_to_disk(os.path.join(out_dir, "unigram"))
    bigram_dataset.save_to_disk(os.path.join(out_dir, "bigram"))
    trigram_dataset.save_to_disk(os.path.join(out_dir, "trigram"))


def make_random_tasks(out_dir, tokenizer_name="EleutherAI/pythia-1b"):
    """
    Generates random binary strings and random integer strings of length 0-60.
    Tokenizes them and saves them as Hugging Face datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    def random_binary_string():
        return " ".join(random.choice("01") for _ in range(2048))

    def random_int_string_60():
        return " ".join(random.choice("0123456789") for _ in range(2048))  # Only digits

    num_tasks = 100000  # 100k examples per dataset

    binary_tasks = [{"text": random_binary_string()} for _ in range(num_tasks)]
    int_string_tasks = [{"text": random_int_string_60()} for _ in range(num_tasks)]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Create and tokenize datasets
    binary_dataset = Dataset.from_list(binary_tasks).map(
        tokenize_function, batched=True
    )
    int_string_dataset = Dataset.from_list(int_string_tasks).map(
        tokenize_function, batched=True
    )

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save datasets to disk
    binary_dataset.save_to_disk(os.path.join(out_dir, "random_binary_dataset"))
    int_string_dataset.save_to_disk(os.path.join(out_dir, "random_int_string_dataset"))

    print(f"Generated and saved datasets in {out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "cache_data_txt": cache_data_txt,
            "cache_data": cache_data,
            "make_dummy_tasks": make_dummy_tasks,
            "make_random_tasks": make_random_tasks,
        }
    )
