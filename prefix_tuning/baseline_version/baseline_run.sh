#!/bin/bash

# disi-unibo-nlp/bc5cdr  disi-unibo-nlp/anatem  disi-unibo-nlp/jnlpba
python baseline_main.py  --dataset disi-unibo-nlp/bc5cdr --learning_rate 1e-4  >> baseline_bc5cdr.txt
python baseline_main.py  --dataset disi-unibo-nlp/anatem --learning_rate 1e-4  >> baseline_anatem.txt
python baseline_main.py  --dataset disi-unibo-nlp/jnlpba --learning_rate 1e-4  >> baseline_jnlpba.txt
