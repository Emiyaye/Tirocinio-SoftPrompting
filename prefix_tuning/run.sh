#!/bin/bash

python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5" >> results/testing_results.txt
python test.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10" >> results/testing_results.txt

python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5" >> results/testing_results.txt
python test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10" >> results/testing_results.txt

python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5" >> results/testing_results.txt
python test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10" >> results/testing_results.txt

#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10"    
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5"
#python training.py --model "medicalai/ClinicalBERT" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10"

#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5"
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10"

#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "1"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "5"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --token_lenght "10"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "1"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "5"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --token_lenght "10"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "1"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "5"
#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --token_lenght "10"

#python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/ncbi" --subset "500"
#python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc4chemd" --subset "500"
#python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc2gm" --subset "500"
#python train_n_test.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --subset "500"
#python train_n_test.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --subset "500"