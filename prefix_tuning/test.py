# Test only
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from seqeval.metrics import classification_report
from tqdm import tqdm
from ner_prefix_tuning_model import NERPrefixTuningModel
from typing import Dict, List
import argparse

parser = argparse.ArgumentParser(description='Valuta un modello NER con Prefix Tuning.')
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='Nome del modello base da Hugging Face.')
parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr',
                    help='Nome del dataset da Hugging Face nel formato "user/dataset".')
args = parser.parse_args()


MODEL_NAME = args.model
DATASET_NAME = args.dataset
PREFIX_LENGTH = 10
MID_DIM = 512
BATCH_SIZE = 8
MAX_SEQ_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE_NAME = MODEL_NAME + "_" + DATASET_NAME.replace("/", "-") + ".pth"

# Carica il dataset da Hugging Face
print("Caricamento del dataset di test...")
dataset_dict = load_dataset(DATASET_NAME)
# Seleziona un subset del dataset di test
test_dataset = dataset_dict['test']

# Estrai i tag NER unici dal dataset di addestramento per coerenza
print("Estrazione dei tag NER unici...")
unique_tags = set()
for example in dataset_dict['train']:
    for tag in example['ner_tags']:
        unique_tags.add(tag)

if 'O' in unique_tags:
    unique_tags.remove('O')
NER_TAGS = ['O'] + sorted(list(unique_tags))
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAGS)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}

print(f"Tag NER trovati: {NER_TAGS}")

# Prepara il tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Funzione per tokenizzare e allineare le etichette
def tokenize_and_align(examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
    tokenized_inputs = tokenizer(
        examples['tokens'],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        is_split_into_words=True
    )
    
    labels = []
    for i, label_list in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                continue
            current_tag = label_list[word_idx]
            label_ids[word_ids.index(word_idx)] = TAG_TO_ID[current_tag]
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

processed_test_dataset = test_dataset.map(tokenize_and_align, batched=True)
processed_test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

test_dataloader = DataLoader(processed_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## TEST

print("\nInizializzazione del modello e caricamento dei pesi...")
model = NERPrefixTuningModel(
    model_name=MODEL_NAME,
    ner_tags=NER_TAGS,
    prefix_length=PREFIX_LENGTH,
    mid_dim=MID_DIM
).to(DEVICE)

model.load_state_dict(torch.load(FILE_NAME, map_location=DEVICE))
model.eval()
print("Modello caricato con successo!")

true_labels = []
predicted_labels = []

print("\nInizio della valutazione...")
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Valutazione"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs, dim=-1)

        for i in range(labels.shape[0]):
            true_tags = []
            pred_tags = []
            
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    true_tags.append(ID_TO_TAG[labels[i, j].item()])
                    pred_tags.append(ID_TO_TAG[predictions[i, j].item()])
            
            true_labels.append(true_tags)
            predicted_labels.append(pred_tags)

## STAMPA RISULTATI

print("\nReport di classificazione (test):")
print(classification_report(true_labels, predicted_labels, digits=4))



print("\n\n")

# Stampa una tabella con parola, tag reale, tag previsto
test_dataloader_five = DataLoader(processed_test_dataset.select(range(5)), batch_size=1, shuffle=False)

with torch.no_grad():
    for i, batch in enumerate(test_dataloader_five):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

        original_tokens = test_dataset[i]['tokens']
        true_tags_original = test_dataset[i]['ner_tags']
        word_ids = tokenizer(original_tokens, is_split_into_words=True).word_ids()
        
        print(f"\n--- Esempio {i+1} ---")
        print("Testo originale:", " ".join(original_tokens))
        print("Parola\t\tTag Reale\tTag Previsto")
        print("--------------------------------------------------")
        
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_tag = ID_TO_TAG[predictions[token_idx]]
                true_tag = true_tags_original[word_idx]
                word = original_tokens[word_idx]
                
                print(f"{word:<15}\t{true_tag:<15}\t{predicted_tag:<15}")
                
            previous_word_idx = word_idx
