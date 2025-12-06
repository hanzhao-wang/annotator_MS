#!/usr/bin/env bash
# 
# run_experiments.sh


accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos0.5-unif.json --seed 42

accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos0.6-unif.json --seed 42

accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos0.7-unif.json --seed 42

accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos0.8-unif.json --seed 42

accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos0.9-unif.json --seed 42

accelerate launch run.py /home/zhongzec24/RewardModeling/paper_experiment_configs/llama-1b-pos1.0-unif.json --seed 42