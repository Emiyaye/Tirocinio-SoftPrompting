#!/bin/bash

MODEL="$1"
DATASET="$2"
PREFIX_LENGTH="$3"
LEARNING_RATE="$4"
OUTPUT_FILE="$5"

COMMAND="python training.py --model \"$MODEL\" --dataset \"$DATASET\""

if [ "$PREFIX_LENGTH" != "0" ] && [ ! -z "$PREFIX_LENGTH" ]; then
    COMMAND+=" --prefix_length \"$PREFIX_LENGTH\""
fi


if [ ! -z "$LEARNING_RATE" ]; then
    COMMAND+=" --learning_rate \"$LEARNING_RATE\""
fi

eval $COMMAND >> "$OUTPUT_FILE"