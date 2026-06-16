import os
import fire
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# C4ローカルサブセット特化版のスクリプト

def cache_data(
    out_dir: str,
    local_dir: str,
    tokenizer_name: str = "EleutherAI/pythia-1b",
):
    """
    ローカルのC4 Arrowファイル（先頭50ファイル）をロードし、並列処理でトークナイズして保存します。
    保存時、ファイル名を 'data-XXXXX-of-01637.arrow' 形式に修正します。

    Args:
        out_dir: トークナイズされたデータの出力ディレクトリ
        local_dir: ローカルのC4 Arrowファイルが保存されているディレクトリパス
        tokenizer_name: 使用するトークナイザー名
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # 並列処理に使うプロセス数を指定
    NUM_PROC = 24  
    
    # --- C4ローカルファイルサブセットのロード ---
    
    # 1. 処理対象の先頭50ファイル (00000-00049) のリストを生成
    num_files_to_load = 50
    total_files_in_source = 1637 # 元データの総ファイル数 (添付画像より)
    target_output_total = 1637    # 出力ファイル名に含める総ファイル数
    
    # 0から49までの5桁のインデックス文字列を作成
    file_indices = [f"{i:05d}" for i in range(num_files_to_load)] 
    
    # ターゲットとなるローカルArrowファイルへの完全パスリストを構築
    # ファイル名形式: c4-train-00000-of-01637.arrow (添付画像より)
    data_files_list = [
        os.path.join(local_dir, f"c4-train-{i}-of-{total_files_in_source:05d}.arrow")  
        for i in file_indices
    ]
    
    print(f"Loading {num_files_to_load} C4 files from local directory: {local_dir}")
    
    # 2. Hugging Face DatasetsのArrowローダーでローカルファイルをロード
    dataset = load_dataset(
        "arrow",                                     
        data_files={"train": data_files_list},       
        split="train",
        streaming=False,
    )
    
    # 3. トークナイズ処理を並列化
    print(f"Tokenizing dataset using {NUM_PROC} processes...")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"], 
            truncation=True, 
            max_length=1024
        ),
        batched=True,
        num_proc=NUM_PROC  
    ).remove_columns(["text"])
    
    # 4. 保存とファイル名修正
    if out_dir is not None:
        # ディレクトリが存在しない場合は作成
        os.makedirs(out_dir, exist_ok=True)
        
        # トークナイズされたデータセットを一旦保存 (シャード数は自動決定される)
        dataset.save_to_disk(out_dir)
        print(f"Tokenized dataset saved to: {out_dir}")
        
        # --- ファイル名修正ロジック ---
        target_total_str = f"{target_output_total:05d}"
        
        print(f"Renaming saved shards to 'data-XXXXX-of-{target_total_str}.arrow' format...")
        
        # 保存されたシャードファイルを検索し、名前を変更
        for filename in os.listdir(out_dir):
            if filename.startswith("data-") and filename.endswith(".arrow"):
                parts = filename.split('-')
                
                # 'data-XXXXX-of-YYYYY.arrow' の形式であることを確認
                if len(parts) == 4 and parts[2] == 'of':
                    current_index_str = parts[1] # e.g., '00000'
                    
                    new_filename = f"data-{current_index_str}-of-{target_total_str}.arrow"
                    
                    old_path = os.path.join(out_dir, filename)
                    new_path = os.path.join(out_dir, new_filename)
                    
                    # ファイル名が異なる場合のみリネームを実行
                    if filename != new_filename:
                        os.rename(old_path, new_path)
                        
        print("File renaming complete.")
        # ---------------------------
    else:
        print("Dataset processed, but out_dir was not provided, so data was not saved.")


if __name__ == "__main__":
    # 実行する関数を cache_data のみに限定
    fire.Fire(cache_data)