#!/bin/bash


python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "1" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "5" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "10" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "20" >> results/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "50" >> results/testing_results.txt


