 
 # PPT w/ MP-STRUCT => PT w/ NL data (C4)
 python train.py --model_name EleutherAI/pythia-1b --data_dir ./data/tokenized/mg_v3_0.2 --output_dir /data1/mamita/output/mg_v3_0.2/pythia-1b --save_steps 500 --max_steps 500 --reinit True 
 python train.py --model_name /data1/mamita/output/mg_v3_0.2/pythia-1b/checkpoint-500 --data_dir /data1/mamita/data/tokenized/c4-1b --output_dir /data2/mamita/output/c4/pythia-1b/mg_v3_0.2 --save_steps 2000 --max_steps 25000 --reinit False




 # PPT w/ MP-STRUCT CORE => PT w/ NL data (C4)
 python train.py --model_name EleutherAI/pythia-1b --data_dir ./data/tokenized/mp_head_constrained_hybrid --output_dir /data1/mamita/output/mp_head_constrained_hybrid/pythia-1b --save_steps 500 --max_steps 500 --reinit True
 python train.py --model_name /data1/mamita/output/mp_head_constrained_hybrid/pythia-1b/checkpoint-500 --data_dir /data1/mamita/data/tokenized/c4-1b --output_dir /data2/mamita/output/c4/pythia-1b/mp_head_constrained_hybrid --save_steps 2000 --max_steps 25000 --reinit False 