# c4 download 
#HF_DATASETS_CACHE=/data1/mamita/hf_cache python src.load_c4_data /data1/mamita/data/c4

# c4 tokenization (c4_max=25)
python -m src.utils-c4 cache_data  --local_dir /data1/mamita/data/c4  --out_dir /data1/mamita/data/tokenized/c4-1b --tokenizer_name "EleutherAI/pythia-1b"


