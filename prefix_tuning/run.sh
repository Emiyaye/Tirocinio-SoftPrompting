#!/bin/bash

# disi-unibo-nlp/bc5cdr  disi-unibo-nlp/anatem  disi-unibo-nlp/jnlpba
MODELS=("dmis-lab/biobert-v1.1" "bert-base-cased" "bert-base-uncased" "medicalai/ClinicalBERT" "roberta-base" "roberta-base-uncased")

for MODEL in "${MODELS[@]}"
do  
    python model_analysis_Fantazzini.py --model "$MODEL" --prefix_length 50 --batch_size 8 --dataset disi-unibo-nlp/bc5cdr
done