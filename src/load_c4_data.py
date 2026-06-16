import os
import fire
from datasets import load_dataset


def load_c4_data(
    out_dir: str = "./data/c4_en_raw",
    config_name: str = "en",
    dataset_name: str = "allenai/c4",
):
    """
    Hugging Face HubからC4データセット全体をロードし、キャッシュします。
    (トークナイズ処理は行いません)

    Args:
        out_dir: (未使用 - 互換性のため残す)
        config_name: C4のコンフィグレーション名 (デフォルトは 'en')
        dataset_name: データセット名 (デフォルトは 'allenai/c4')
    """
    
    # 並列処理に使うプロセス数を指定 (ここではロード処理自体に直接は適用されないが、
    # load_dataset内部のダウンロード処理等に影響を与える可能性がある)
    NUM_PROC = 24  
    
    print(f"Loading full C4 ('{config_name}' config).")
    
    # C4全体をロード (data_files引数なし)
    # **注意: Gated Dataのため、事前にhuggingface-cli loginとアクセス承認が必要です。**
    
    # ここでデータセットがダウンロードされ、HF_DATASETS_CACHEにキャッシュされます
    dataset = load_dataset(
        dataset_name, 
        config_name,                               
        split="train",
    )
    
    print("C4 dataset loaded and cached successfully (no tokenization performed).")
    
    # ロードしたデータセットのメタ情報を表示
    print(f"Dataset size: {len(dataset)} examples.")
    print(f"First example keys: {list(dataset[0].keys())}")
    
    # トークナイズ処理やsave_to_diskは行わない


if __name__ == "__main__":
    # このスクリプトを直接実行できるようにFireを設定
    fire.Fire(load_c4_data)