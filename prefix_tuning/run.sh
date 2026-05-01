#!/bin/bash

# 
MODELS=("bert-base-cased" "bert-base-uncased" "medicalai/ClinicalBERT" "roberta-base" "roberta-base-uncased")

for MODEL in "${MODELS[@]}"
do  
    python main.py --model "$MODEL" --prefix_length 80 --num_epochs 50 --batch_size 16 --dataset disi-unibo-nlp/jnlpba
done