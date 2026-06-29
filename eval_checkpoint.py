from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import datasets

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import fire
import glob


def id_estimate(X, fraction=0.9, verbose=False):
    Y = np.sort(X, axis=1, kind="quicksort")

    k1 = Y[:, 1]
    k2 = Y[:, 2]

    zeros = np.where(k1 == 0)[0]
    degeneracies = np.where(k1 == k2)[0]

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print(f"Fraction good points: {good.shape[0] / Y.shape[0]}")

    k1 = k1[good]
    k2 = k2[good]

    npoints = int(np.floor(good.shape[0] * fraction))

    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind="quicksort")
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    x_good = x[0:npoints, np.newaxis]
    y_good = y[0:npoints, np.newaxis]
    not_nan_idx = np.logical_not(np.isnan(x_good))
    x_good = x_good[not_nan_idx][:, np.newaxis]
    y_good = y_good[not_nan_idx][:, np.newaxis]

    regr = LinearRegression(fit_intercept=False)
    regr.fit(x_good, y_good)

    r, pval = pearsonr(x_good.reshape(x_good.shape[0]), y_good.reshape(y_good.shape[0]))
    return x, y, regr.coef_[0][0], r, pval


def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    a_squared_norm = torch.sum(a**2, dim=-1, keepdim=True).clamp(min=eps)
    b_squared_norm = torch.sum(b**2, dim=-1, keepdim=True).clamp(min=eps)

    dot_product = torch.matmul(a, b.transpose(-2, -1))

    similarity = dot_product / torch.sqrt(
        a_squared_norm * b_squared_norm.transpose(-2, -1)
    )

    similarity = torch.clamp(similarity, min=-1.0, max=1.0)

    return similarity


def convert_np_to_py_type(x):
    if type(x).__module__ == "numpy":
        return x.item()
    else:
        return x


def _get_num_layers(model_dir):
    try:
        config = AutoConfig.from_pretrained(model_dir)
        return config.num_hidden_layers
    except Exception:
        return 16


def compute_stats(
    model_dir,
    compute_loss=True,
    compute_weight_norm=True,
    compute_id=True,
    sample_size=1024,
    bsz=1,
    dataset=None,
    dataset_name=None,
    tokenizer_name="EleutherAI/pythia-1b",
    max_length=1024,
):
    num_layers = _get_num_layers(model_dir)

    id_output_path = os.path.join(model_dir, "id.json")
    if os.path.exists(id_output_path):
        print(f"{id_output_path} already exists.")
        compute_id = False

    weight_norm_output_path = os.path.join(model_dir, "weight_norm.json")
    if os.path.exists(weight_norm_output_path):
        print(f"{weight_norm_output_path} already exists.")
        compute_weight_norm = False

    output_bool_vals = {
        "loss": compute_loss,
        "intrinsic dimension": compute_id,
        "weight_norms": compute_weight_norm,
    }
    if not compute_loss and not compute_id and not compute_weight_norm:
        print("Not computing loss, intrinsic dimension, or weight norm. Returning...")
        return
    else:
        print(f"Computing {', '.join([k for k, v in output_bool_vals.items() if v])}.")

    if dataset is None:
        if dataset_name == "c4":
            dataset = (
                datasets.load_dataset("allenai/c4", "en", split="validation")
                .select(range(sample_size))
            )
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Use 'c4'.")

    model = AutoModelForCausalLM.from_pretrained(model_dir).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    def tokenize_examples(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    def get_lengths_and_labels(examples):
        input_ids = examples["input_ids"]
        attention_mask = examples["attention_mask"]

        labels = []
        lengths = []

        for i in range(len(input_ids)):
            current_input_ids = input_ids[i]
            current_attention_mask = attention_mask[i]

            current_labels = [
                -100 if token == tokenizer.pad_token_id else token
                for token in current_input_ids
            ]
            labels.append(current_labels)

            length = sum(current_attention_mask) - 1
            lengths.append(length)

        examples["labels"] = labels
        examples["lengths"] = lengths
        return examples

    collated = dataset.map(tokenize_examples, batched=True).map(
        get_lengths_and_labels, batched=True
    )

    model_kwargs = {
        "return_dict": True,
    }
    if compute_id:
        model_kwargs["output_hidden_states"] = True

    @torch.inference_mode()
    def compute_outputs(
        model, model_kwargs, input_ids, attention_mask, labels=None, lengths=None
    ):
        output_dict = {}
        if compute_loss:
            model_kwargs["labels"] = torch.tensor(labels).cuda()

        outputs = model(
            input_ids=torch.tensor(input_ids).cuda(),
            attention_mask=torch.tensor(attention_mask).cuda(),
            **model_kwargs,
        )

        if compute_weight_norm:
            param_weights = []
            for name, param in model.named_parameters():
                if param.data is not None:
                    param_weights.append(param.data.cpu().detach().numpy().flatten())
            output_dict["weight_norms"] = [
                (np.linalg.norm(np.concatenate(param_weights)) ** 2).item()
            ]

        if compute_loss:
            if isinstance(outputs, dict):
                output_dict["loss"] = [outputs["loss"].item()]
            else:
                output_dict["loss"] = [outputs.loss.item()]

        if compute_id:
            lengths = torch.tensor(lengths).cuda()
            if isinstance(outputs, dict):
                for layer in range(num_layers):
                    output_dict[f"cls_embeddings_{layer}"] = [
                        outputs["hidden_states"][layer][
                            torch.arange(len(lengths)), lengths, :
                        ]
                    ]
            else:
                for layer in range(num_layers):
                    output_dict[f"cls_embeddings_{layer}"] = [
                        outputs.hidden_states[layer][
                            torch.arange(len(lengths)), lengths, :
                        ]
                    ]
        return output_dict

    cols = collated.column_names
    collated = collated.map(
        lambda ex: compute_outputs(
            model,
            model_kwargs,
            ex["input_ids"],
            ex["attention_mask"],
            ex["labels"],
            ex["lengths"],
        ),
        batched=True,
        batch_size=bsz,
        remove_columns=cols,
    )

    if compute_id:
        ret = {}
        for layer in range(num_layers):
            cls_embeddings = list(
                [torch.FloatTensor(x) for x in collated[f"cls_embeddings_{layer}"]]
            )
            cls_embeddings = torch.cat(cls_embeddings, dim=0)
            emb_sim_matrix = (
                sim_matrix(cls_embeddings, cls_embeddings).cpu().detach().numpy()
            )
            _, _, d, r, pval = id_estimate(
                1 - emb_sim_matrix, fraction=1.0, verbose=False
            )
            print(
                f"Layer: {layer}. Intrinsic dimension: {d}, Pearson corr coeff: {r}, p-value: {pval}"
            )
            ret[layer] = {
                "intrinsic_dimension": convert_np_to_py_type(d),
                "pearson_corr_coeff": convert_np_to_py_type(r),
                "p-value": convert_np_to_py_type(pval),
            }

        with open(id_output_path, "w") as f:
            json.dump(ret, f)

    if compute_loss:
        loss = list(collated["loss"])
        with open(
            os.path.join(model_dir, f"loss_{dataset_name}.json"),
            "w",
        ) as f:
            f.write(json.dumps(loss) + "\n")

    if compute_weight_norm:
        param_weights = []
        for _, param in model.named_parameters():
            param_weights.append(param.data.cpu().detach().numpy().flatten())
        weight_norms = (np.linalg.norm(np.concatenate(param_weights)) ** 2).item()
        with open(weight_norm_output_path, "w") as f:
            f.write(json.dumps(weight_norms) + "\n")


def blimp_eval(
    model_dir,
    bsz=128,
    dataset=None,
    tokenizer_name="EleutherAI/pythia-1b",
):
    if dataset is None:
        dataset = datasets.load_dataset("WillHeld/blimp")["train"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model = AutoModelForCausalLM.from_pretrained(model_dir).cuda()

    def tokenize_examples(examples):
        good = tokenizer(
            examples["sentence_good"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        bad = tokenizer(
            examples["sentence_bad"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        return {
            "good_input_ids": good["input_ids"],
            "good_attention_mask": good["attention_mask"],
            "bad_input_ids": bad["input_ids"],
            "bad_attention_mask": bad["attention_mask"],
        }

    tokenized = dataset.map(tokenize_examples, batched=True)

    model_kwargs = {
        "return_dict": True,
    }

    @torch.inference_mode()
    def compute_batch_accuracy(
        model,
        model_kwargs,
        good_input_ids,
        good_attention_mask,
        bad_input_ids,
        bad_attention_mask,
    ):
        good_input_ids = torch.tensor(good_input_ids).cuda()
        good_attention_mask = torch.tensor(good_attention_mask).cuda()
        bad_input_ids = torch.tensor(bad_input_ids).cuda()
        bad_attention_mask = torch.tensor(bad_attention_mask).cuda()
        good_output = model(
            input_ids=good_input_ids,
            attention_mask=good_attention_mask,
            **model_kwargs,
        )
        bad_output = model(
            input_ids=bad_input_ids,
            attention_mask=bad_attention_mask,
            **model_kwargs,
        )
        good_ll = torch.tensor(
            [
                -F.cross_entropy(
                    good_output.logits[i, : length - 1],
                    good_input_ids[i, 1:length],
                    reduction="sum",
                )
                for i, length in enumerate(
                    (good_input_ids != tokenizer.pad_token_id).sum(1)
                )
            ]
        )

        bad_ll = torch.tensor(
            [
                -F.cross_entropy(
                    bad_output.logits[i, : length - 1],
                    bad_input_ids[i, 1:length],
                    reduction="sum",
                )
                for i, length in enumerate(
                    (bad_input_ids != tokenizer.pad_token_id).sum(1)
                )
            ]
        )
        is_correct = good_ll > bad_ll
        return {"correct": is_correct.cpu().numpy()}

    cols = tokenized.column_names
    tokenized = tokenized.map(
        lambda ex: compute_batch_accuracy(
            model,
            model_kwargs,
            ex["good_input_ids"],
            ex["good_attention_mask"],
            ex["bad_input_ids"],
            ex["bad_attention_mask"],
        ),
        batched=True,
        batch_size=bsz,
        remove_columns=cols,
    )

    num_examples = len(dataset)
    correct = np.sum(list(tokenized["correct"])) / num_examples
    correct = convert_np_to_py_type(correct)

    with open(os.path.join(model_dir, "blimp.json"), "w") as f:
        f.write(json.dumps(correct) + "\n")

    with open(os.path.join(model_dir, "blimp_preds.json"), "w") as f:
        f.write(json.dumps(tokenized["correct"]) + "\n")

    print(f"BLiMP accuracy: {correct:.4f}")


def main(
    super_dir,
    bsz=32,
    sample_size=1024,
    compute_id=False,
    do_eval=True,
    do_blimp=True,
    tokenizer_name="EleutherAI/pythia-1b",
):
    if do_eval:
        val_dataset = (
            datasets.load_dataset("allenai/c4", "en", split="validation")
            .select(range(sample_size))
        )
    if do_blimp:
        blimp_dataset = datasets.load_dataset("WillHeld/blimp")["train"]

    for model_dir in sorted(glob.glob(super_dir + "/*")):
        if not os.path.isdir(model_dir):
            continue
        print(f"Processing {model_dir}")
        if do_eval:
            compute_stats(
                model_dir,
                bsz=bsz,
                sample_size=sample_size,
                dataset=val_dataset,
                compute_id=compute_id,
                tokenizer_name=tokenizer_name,
            )
        if do_blimp:
            blimp_eval(
                model_dir,
                bsz=bsz,
                dataset=blimp_dataset,
                tokenizer_name=tokenizer_name,
            )


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "blimp": blimp_eval,
            "compute_stats": compute_stats,
        }
    )
