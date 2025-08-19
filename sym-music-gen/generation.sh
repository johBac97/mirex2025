#!/bin/bash

input_json_prompt=$1
output_folder=$2
number_generations=$3

model=runs/small_llama_pretokenized_real_2_cont_2/checkpoint-454500/

uv run src/generate.py \
    $model \
    $input_json_prompt \
    --output $output_folder \
    --num-generations $number_generations
