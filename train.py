from typing import List

import datasets
import fire
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


class SaveAtStepsCallback(TrainerCallback):
    def __init__(self, save_steps: List[int], output_dir: str):
        self.save_steps = sorted(save_steps)
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)
            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            print(f"Saved model at step {state.global_step}")


def main(
    data_dir="./data/tokenized/shuff_dyck",
    model_name="EleutherAI/pythia-1b",
    reinit=False,
    max_seq_length=1024,
    gradient_accumulation_steps=2,
    max_steps=25000,
    bsz=16,
    warmup_steps=1000,
    logging_steps=5,
    save_steps=500,
    output_dir="output",
    seed=3407,
    report_to="wandb",
    lr=5e-4,
    min_lr_rate=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
    override_packing=False,
    use_callback=False,
):
    print(locals())
    set_seed(seed)

    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 100)) + list(range(4000, 26000, 1000)),
        output_dir=output_dir,
    )

    dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    if reinit:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    packing = "c4" in data_dir

    training_args = SFTConfig(
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        seed=seed,
        report_to=report_to,
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": min_lr_rate},
        packing=packing if not override_packing else False,
        max_length=max_seq_length,
        bf16=True,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    if use_callback:
        trainer.add_callback(callback)

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
