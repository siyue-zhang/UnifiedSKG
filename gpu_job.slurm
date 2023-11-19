#!/bin/bash
#SBATCH --job-name=t5_base_wtq
#SBATCH --open-mode=append
#SBATCH --output=./t5_base_wtq.out
#SBATCH --error=./t5_base_wtq.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay /scratch/sz4651/Projects/overlay-share.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
conda activate /ext3/miniconda3/envs/py3.7pytorch1.8new
cd /scratch/sz4651/Projects/UnifiedSKG
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_base_finetune_wikitq.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 100 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
"