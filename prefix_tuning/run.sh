#!/bin/bash


python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-3" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-4" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-5" >> results/testing_results.txt
