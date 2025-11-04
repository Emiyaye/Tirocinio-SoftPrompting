#!/bin/bash

RUNNER_SCRIPT="./run.sh"

MODELS=("medicalai/ClinicalBERT" )
DATASETS=("disi-unibo-nlp/AnatEM" "disi-unibo-nlp/JNLPBA" "disi-unibo-nlp/bc5cdr")
PREFIX_LENGTHS=("10" "20" "50")
LEARNING_RATES=("1e-3" "1e-4" "1e-5")
OUTPUT_FILE="results/prova/soft_testing_results.txt"


for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for PREFIX_LENGTH in "${PREFIX_LENGTHS[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
                # Call the runner script: 
                # $1 MODEL $2 DATASET $3 PREFIX_LENGTH $4 LEARNING_RATE $5 OUTPUT_FILE
                "$RUNNER_SCRIPT" "$MODEL" "$DATASET" "$PREFIX_LENGTH" "$LR" "$OUTPUT_FILE"
            done
        done
    done
done