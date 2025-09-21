Per fare un training:
    python train_n_test.py --model "$MODEL_NAME" --dataset "$DATASET_NAME" --subset "$SUBSET_SIZE" --token_lenght "$LENGHT"

Per fare solo un test:
    python test.py --model "$MODEL_NAME" --dataset "$DATASET_NAME" --token_lenght "$LENGHT"

modelli testati:
bert-base-uncased
dmis-lab/biobert-v1.1

datasets:
disi-unibo-nlp/JNLPBA
disi-unibo-nlp/ncbi
disi-unibo-nlp/bc4chemd
disi-unibo-nlp/bc2gm
disi-unibo-nlp/AnatEM
disi-unibo-nlp/bc5cdr
