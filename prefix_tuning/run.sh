#!/bin/bash

python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --subset "500"
python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/ncbi" --subset "500"
python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc4chemd" --subset "500"
python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc2gm" --subset "500"
python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --subset "500"
python train_n_test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --subset "500"