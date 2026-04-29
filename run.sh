#!/bin/bash

# 
MODELS=("dmis-lab/biobert-v1.1" "bert-base-cased" "bert-base-uncased" "medicalai/ClinicalBERT" "roberta-base" "roberta-base-uncased")

for MODEL in "${MODELS[@]}"
do  
    python main.py --model "$MODEL" --prefix_length 50 --num_epochs 50 --batch_size 16
done