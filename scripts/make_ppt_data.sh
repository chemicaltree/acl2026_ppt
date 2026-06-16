 # MP-STRUCT
 python -m src.minimalist_grammar_v3 generate_mg_sentences --file_dir ./data/mg_v3_0.2  --n 100000  -max_length 1024 -strip_lexical True --p_c_wh 0.2 --p_wh_goal 0.2
 python -m src.utils cache_data --dataset_name data/mg_v3_0.2/mg_ids_100000_1024.txt  --out_dir ./data/tokenized/mg_v3_0.2


 # MP-STRUCT CORE
 python -m src.generate_enhanced_constrained --n 100000 --k_struct 1 --k_dep 4  --out_dir ./data/mp_head_constrained_hybrid --use_head_diversity=True --use_complex_args=False
 python -m src.utils cache_data --dataset_name data/mp_head_constrained_hybrid/hybrid_ids.txt --out_dir ./data/tokenized/mp_head_constrained_hybrid