import os
import random

import fire
import numpy as np
from tqdm import trange


def generate_dyck(num_symbols, min_depth=1, max_depth=4, max_length=510, offset=None):
    result = []
    stack = []

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    if offset is None:
        offset = num_symbols

    for _ in range(min_depth):
        opening_symbol = np.random.randint(0, num_symbols)
        result.append(opening_symbol)
        stack.append(opening_symbol)

    while len(result) < max_length:
        if (
            len(stack) < max_depth and random.random() < 0.5
        ):
            if len(result) >= max_length - 1:
                closing_symbol = stack.pop() + offset
                result.append(closing_symbol)
                continue
            opening_symbol = np.random.randint(0, num_symbols)
            result.append(opening_symbol)
            stack.append(opening_symbol)
        else:
            closing_symbol = stack.pop() + offset
            result.append(closing_symbol)
            if not stack:
                break

    while stack:
        closing_symbol = stack.pop() + offset
        result.append(closing_symbol)

    return result if not stack else None


def generate_dyck_txt_file(
    file_dir, num_symbols=1, n=100000, target_length=1024, min_depth=1, max_depth=16
):
    os.makedirs(file_dir, exist_ok=True)
    with open(
        f"{file_dir}/dyck_sequences_{num_symbols}_{min_depth}_{max_depth}.txt", "w"
    ) as f:
        for i in trange(n):
            result = []
            while len(result) < target_length:
                new_seq = generate_dyck(
                    num_symbols, min_depth=min_depth, max_depth=max_depth
                )
                if new_seq is None:
                    continue
                result.extend(new_seq)

            dyck_str = " ".join(
                [str(x) for x in result[:target_length]]
            )
            f.write(f"{dyck_str}\n")


def make_copy_tokens(
    num_symbols: int = 64, min_w_length: int = 10, max_w_length: int = 510
):
    if min_w_length > max_w_length:
        raise ValueError("min_w_length cannot be greater than max_w_length")

    length = random.randint(min_w_length, max_w_length)
    original_token_seq = np.random.randint(0, num_symbols, size=length).tolist()
    return original_token_seq + original_token_seq


def make_copy_str_file(
    file_dir,
    num_symbols: int = 64,
    n: int = 100000,
    seq_length: int = 1024,
    min_w_length: int = 10,
):
    os.makedirs(file_dir, exist_ok=True)
    with open(f"{file_dir}/ww_sequences_{num_symbols}_{min_w_length}.txt", "w") as f:
        for _ in trange(n):
            sequence = make_copy_tokens(num_symbols, min_w_length=min_w_length)
            while len(sequence) < seq_length:
                sequence.extend(
                    make_copy_tokens(num_symbols, min_w_length=min_w_length)
                )
            repeated_str = " ".join(map(str, sequence[:seq_length]))
            f.write(f"{repeated_str}\n")


def generate_shuff_dyck(k, max_length=1024, p_open=0.5, min_depth=1, max_depth=8):
    sequence = []
    counts = [0] * k

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    for _ in range(min_depth):
        bracket = random.randint(0, k - 1)
        sequence.append(bracket)
        counts[bracket] += 1

    while len(sequence) < max_length:
        depth = sum(counts)

        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_shuff_dyck_txt_file(
    file_dir, num_symbols=64, n=100000, target_length=1024, p=0.51
):
    os.makedirs(file_dir, exist_ok=True)
    with open(
        f"{file_dir}/dyck_sequences_cross_serial_{num_symbols}_{p}.txt", "w"
    ) as f:
        for i in range(n):
            result = generate_shuff_dyck(num_symbols, target_length, p)
            dyck_str = " ".join([str(x) for x in result[:target_length]])
            f.write(f"{dyck_str}\n")


def generate_random_baseline(
    file_dir, vocab_size=128, n=100000, target_length=1024, seed=42
):
    rng = random.Random(seed)
    os.makedirs(file_dir, exist_ok=True)
    with open(f"{file_dir}/random_sequences_{vocab_size}.txt", "w") as f:
        for _ in trange(n):
            seq = [str(rng.randint(0, vocab_size - 1)) for _ in range(target_length)]
            f.write(" ".join(seq) + "\n")


if __name__ == "__main__":
    fire.Fire(
        {
            "generate_dyck": generate_dyck_txt_file,
            "generate_shuff_dyck": generate_shuff_dyck_txt_file,
            "generate_ww": make_copy_str_file,
            "generate_random": generate_random_baseline,
        }
    )
