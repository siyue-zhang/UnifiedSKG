#!/bin/bash
#SBATCH --job-name=Omnitab
#SBATCH --open-mode=append
#SBATCH --output=./monitor/Omnitab_large_finetune_wikitq.out
#SBATCH --error=./monitor/Omnitab_large_finetune_wikitq.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -c 2

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/sz4651/Projects/UnifiedSKG \
&& source ~/.bashrc \
&& source activate /scratch/sz4651/miniconda3/envs/py3.7pytorch1.8new \
&& python -m torch.distributed.launch --nproc_per_node 2 \
--master_port 1234 train.py --seed 2 \
--cfg Salesforce/_Omnitab_large_finetune_wikitq.cfg \
--run_name Omnitab_large_finetune_wikitq \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 50 \
--metric_for_best_model avr \
--greater_is_better true \
--save_strategy steps \
--save_steps 500 \
--save_total_limit 1 \
--load_best_model_at_end \
--gradient_accumulation_steps 32 \
--num_train_epochs 400 \
--adafactor true --learning_rate 5e-5 \
--do_train --do_eval --do_predict --predict_with_generate \
--output_dir output/Omnitab_large_finetune_wikitq \
--overwrite_output_dir \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 8 \
--generation_num_beams 4 \
--generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
"