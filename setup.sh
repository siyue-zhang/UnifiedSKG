#! /bin/bash

# export WANDB_API_KEY=3b48e9a5063a5c5906e7bde4fef1cac8aeeb8aad
# export WANDB_PROJECT=skg

# wandb.init(project='skg',
# 			name=args.exp_name,
# 			config=args)

# srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=32GB --time=20:00:00 --gres=gpu:1 --pty /bin/bash

# singularity exec --nv --overlay /scratch/sz4651/Projects/overlay-share.ext3:rw /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash
singularity exec --nv --overlay overlay-skg.ext3:rw /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash

# source activate /scratch/sz4651/miniconda3/envs/py3.7pytorch1.8new

# WTQ
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_base_finetune_wikitq.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_base_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
# # Spider
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train.py --seed 2 --cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_large_finetune_spider_with_cell_value --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train.py --seed 2 --cfg Salesforce/_T5_large_finetune_spider_tableqa.cfg --run_name T5_base_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/_T5_large_finetune_spider_tableqa --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true

# # Squall
node ./third_party/squall/eval/evaluator.js

# python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train.py --seed 2 --cfg Salesforce/_T5_large_finetune_squall.cfg --run_name T5_large_finetune_squall --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 5 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 400 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/T5_large_finetune_squall --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --generation_num_beams 4 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true

# # Squall
python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 
python train.py \
  --seed 42 \
  --cfg Salesforce/_T5_large_finetune_squall.cfg \
  --run_name T5_large_finetune_squall2 \
  --warmup_ratio 0.1 \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 30 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 600 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 50 \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir output/T5_large_finetune_squall2 \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --max_train_samples 5000 \
  --max_eval_samples 1000

  --ddp_find_unused_parameters true 
  --adafactor true \
 

python train.py \
  --seed 42 \
  --cfg Salesforce/_T5_large_finetune_squall.cfg \
  --load_weights_from output/T5_large_finetune_squall2/checkpoint-1000 \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --metric_for_best_model avr \
  --output_dir output/T5_large_finetune_squall2 \
  --per_device_eval_batch_size 8 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024  \
  --max_train_samples 10 \
  --max_eval_samples 1000
 


# squall tableqa train
python ./train.py \
  --seed 42 \
  --cfg Salesforce/_Omnitab_large_finetune_squall_tableqa.cfg \
  --warmup_ratio 0.1 \
  --max_train_samples 100 \
  --max_eval_samples 10 \
  --num_train_epochs 50 \
  --run_name Omnitab_large_finetune_squall_tableqa \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 2 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 24 \
  --adafactor true \
  --learning_rate 2e-5 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir output/Omnitab_large_finetune_squall_tableqa \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --ddp_find_unused_parameters true

# squall tableqa inference
python ./train.py \
  --seed 42 \
  --cfg Salesforce/_Omnitab_large_finetune_squall_tableqa.cfg \
  --resume_from_checkpoint output/Omnitab_large_finetune_squall_tableqa2/checkpoint-4700 \
  --run_name Omnitab_large_finetune_squall_tableqa2 \
  --do_eval \
  --do_predict \
  --metric_for_best_model avr \
  --predict_with_generate \
  --output_dir output/Omnitab_large_finetune_squall_tableqa2 \
  --per_device_eval_batch_size 8 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024

# WTQ TAPEX nail
python ./train.py --seed 2 --cfg Salesforce/_Omnitab_large_finetune_wikitq.cfg --warmup_ratio 0.1 --num_train_epochs 50 --run_name Omnitab_large_finetune_wikitq --logging_strategy steps --logging_first_step true --logging_steps 10 --evaluation_strategy steps --eval_steps 100 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 24 --adafactor true --learning_rate 2e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/Omnitab_large_finetune_wikitq --overwrite_output_dir --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --generation_num_beams 5 --generation_max_length 128 --input_max_length 1024 --ddp_find_unused_parameters true

python ./train.py \
  --seed 42 \
  --cfg Salesforce/_Omnitab_large_finetune_wikitq.cfg \
  --warmup_ratio 0.1 \
  --max_train_samples 100 \
  --max_eval_samples 10 \
  --num_train_epochs 50 \
  --run_name Omnitab_large_finetune_wikitq \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 2 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 24 \
  --adafactor true \
  --learning_rate 2e-5 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir output/Omnitab_large_finetune_wikitq \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --ddp_find_unused_parameters true



# spider tableqa
python ./train.py \
  --seed 42 \
  --cfg Salesforce/_T5_large_finetune_spider_tableqa.cfg \
  --warmup_ratio 0.1 \
  --num_train_epochs 50 \
  --run_name T5_large_finetune_spider_tableqa \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 24 \
  --adafactor true \
  --learning_rate 2e-5 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir output/T5_large_finetune_spider_tableqa \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --generation_num_beams 5 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --ddp_find_unused_parameters true \
  --max_train_samples 5000 \
  --max_eval_samples 2000 