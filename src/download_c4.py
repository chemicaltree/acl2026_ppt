import os
import fire
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_c4_hub(
    out_dir: str = "./data/c4_en_raw",
    config_name: str = "en",
):
    print(f"Loading full C4 ('{config_name}' config) via HuggingFace datasets API.")
    print("Note: This requires `huggingface-cli login` and access approval for allenai/c4.")

    dataset = load_dataset(
        "allenai/c4",
        config_name,
        split="train",
    )

    print("C4 dataset loaded and cached successfully.")
    print(f"Dataset size: {len(dataset)} examples.")


def download_c4_subset(
    out_dir: str = "./data/c4_en_subset",
    config_name: str = "en",
    split_name: str = "train",
    num_files: int = 50,
):
    os.makedirs(out_dir, exist_ok=True)

    total_files = 1632
    file_indices = [f"{i:05d}" for i in range(num_files)]

    print(f"Starting download of {num_files} files from allenai/c4/{config_name}/{split_name}...")

    for index_str in tqdm(file_indices, desc="Downloading C4 files"):
        filename = f"c4-{split_name}.{index_str}-of-{total_files:05d}.arrow"
        repo_file_path = f"{config_name}/{filename}"

        try:
            hf_hub_download(
                repo_id="allenai/c4",
                filename=repo_file_path,
                local_dir=out_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Error downloading {repo_file_path}: {e}")

    print(f"\nDownload complete. Files saved to: {out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "download_hub": download_c4_hub,
            "download_subset": download_c4_subset,
        }
    )
