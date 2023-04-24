#!/bin/bash
set -x
# BASE_MODEL="/home/vmagent/app/data/gpt-neo-2.7B"
# LORA_PATH="/home/vmagent/app/data/zy_models/lora-gpt-neo-2.7B-alpaca"

# BASE_MODEL="/home/vmagent/app/data/llama-7b-hf,/home/vmagent/app/data/vicuna-7b"
# LORA_PATH="/home/vmagent/app/data/Chinese-Vicuna-lora-7b-belle-and-guanaco,"
# python app.py --model_path_list $BASE_MODEL --lora_path_list $LORA_PATH \
#     --load_8bit "1,0"


BASE_MODEL="/home/vmagent/app/data/vicuna-7b,/home/vmagent/app/data/alpaca-lora-7b"
python app.py --model_path_list $BASE_MODEL \
    --load_8bit "0,0" \
    --use_typewriter 1

# BASE_MODEL="/home/vmagent/app/data/vicuna-7b"
# python app.py --model_path_list $BASE_MODEL \
#     --load_8bit "0"



