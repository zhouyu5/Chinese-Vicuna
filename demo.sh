#!/bin/bash
set -x
# BASE_MODEL="/home/vmagent/app/data/gpt-neo-2.7B"
# LORA_PATH="/home/vmagent/app/data/zy_models/lora-gpt-neo-2.7B-alpaca"
BASE_MODEL="/home/vmagent/app/data/llama-7b-hf"
LORA_PATH="/home/vmagent/app/data/Chinese-Vicuna-lora-7b-belle-and-guanaco"

python app.py --model_path $BASE_MODEL --lora_path $LORA_PATH 
