import os
import fire
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_c4_subset(
    repo_id: str = "allenai/c4",
    config_name: str = "en",
    split_name: str = "train",
    num_files: int = 50, # ダウンロードしたいファイル数
    out_dir: str = "/data1/mamita/c4_en_subset",
):
    """
    Hugging Face HubからC4データセットの特定のArrowファイルをローカルにダウンロードします。
    """
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(out_dir, exist_ok=True)

    # ダウンロードするファイル名のインデックス (0からnum_files-1まで)
    # C4ファイルは5桁のゼロパディングを使用
    file_indices = [f"{i:05d}" for i in range(num_files)] 
    
    # 総ファイル数は1632と固定されているため、これをファイル名パターンに使用
    total_files = 1632 

    print(f"Starting download of {num_files} files from {repo_id}/{config_name}/{split_name}...")
    
    for index_str in tqdm(file_indices, desc="Downloading C4 files"):
        # ファイル名パターンを構築: 例: c4-train.00000-of-01632.arrow
        filename = f"c4-{split_name}.{index_str}-of-{total_files:05d}.arrow"
        
        # リポジトリ内の相対パスを構築: 例: en/c4-train.00000-of-01632.arrow
        repo_file_path = f"{config_name}/{filename}"

        try:
            # hf_hub_downloadを使用してファイルを直接ダウンロード
            hf_hub_download(
                repo_id=repo_id,
                filename=repo_file_path,
                local_dir=out_dir,        # ダウンロード先のローカルディレクトリ
                local_dir_use_symlinks=False, # 確実にコピーを保存
            )
        except Exception as e:
            print(f"Error downloading {repo_file_path}: {e}")
            
    print(f"\nDownload complete. Files saved to: {out_dir}")


if __name__ == "__main__":
    # fireで公開するコマンド名を 'download_c4_subset' とする
    fire.Fire(
        {
            "download_c4_subset": download_c4_subset,
        }
    )

    # python download_c4_subset.py download_c4_subset --out_dir ./data/c4_en_subset
