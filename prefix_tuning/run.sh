#!/bin/bash

python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "10" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "20" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "50" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "10" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "20" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "50" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "10" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "20" >> results/token_lenght/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "50" >> results/token_lenght/testing_results.txt

python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-3" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-AnatEM_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-4" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-AnatEM_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-5" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-AnatEM_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-3" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-JNLPBA_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-4" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-JNLPBA_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-5" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-JNLPBA_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-3" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-bc5cdr_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-4" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-bc5cdr_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-5" --file_name "dmis-lab-biobert-v1.1_disi-unibo-nlp-bc5cdr_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt

python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-3" --file_name "bert-base-uncased_disi-unibo-nlp-AnatEM_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-4" --file_name "bert-base-uncased_disi-unibo-nlp-AnatEM_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --learning_rate "1e-5" --file_name "bert-base-uncased_disi-unibo-nlp-AnatEM_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-3" --file_name "bert-base-uncased_disi-unibo-nlp-JNLPBA_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-4" --file_name "bert-base-uncased_disi-unibo-nlp-JNLPBA_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/JNLPBA" --learning_rate "1e-5" --file_name "bert-base-uncased_disi-unibo-nlp-JNLPBA_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-3" --file_name "bert-base-uncased_disi-unibo-nlp-bc5cdr_learning_rate-1e-3.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-4" --file_name "bert-base-uncased_disi-unibo-nlp-bc5cdr_learning_rate-1e-4.pth" >> results/learning_rate/testing_results.txt
python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/bc5cdr" --learning_rate "1e-5" --file_name "bert-base-uncased_disi-unibo-nlp-bc5cdr_learning_rate-1e-5.pth" >> results/learning_rate/testing_results.txt

#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "10" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "20" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "50" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "10" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "20" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/JNLPBA" --prefix_length "50" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "10" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "20" >> results/token_lenght/testing_results.txt
#python training.py --model "dmis-lab/biobert-v1.1" --dataset "disi-unibo-nlp/bc5cdr" --prefix_length "50" >> results/token_lenght/testing_results.txt
#



#python training.py --model "bert-base-uncased" --dataset "disi-unibo-nlp/AnatEM" --prefix_length "1" >> results/token_lenght/testing_results.txt




