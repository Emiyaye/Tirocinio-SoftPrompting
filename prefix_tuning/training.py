import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from typing import List, Dict, Tuple
from ner_prefix_tuning_model import NERPrefixTuningModel
from datasets import load_dataset
from tqdm import tqdm
from seqeval.metrics import classification_report
import argparse

parser = argparse.ArgumentParser(description='Addestra e valuta un modello NER con Prefix Tuning.')
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='Nome del modello base da Hugging Face.')
parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr',
                    help='Nome del dataset da Hugging Face nel formato "user/dataset".')
parser.add_argument('--token_lenght', type=int, default=10,
                    help='Lunghezza del token')
parser.add_argument('--subset', type=int, default=-1,
                    help='Dimensione del subset di dati da usare per l\'addestramento e il test.')
args = parser.parse_args()

## PREPARAZIONE DATI E HYPERPARAMETRI

MODEL_NAME = args.model
DATASET_NAME = args.dataset
PREFIX_LENGTH = args.token_lenght
MID_DIM = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
MAX_SEQ_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE_NAME = MODEL_NAME.replace("/", "-") + "_" + DATASET_NAME.replace("/", "-") + "_token-lenght-" + str(PREFIX_LENGTH) + ".pth"

# Carica il dataset da Hugging Face
print(f"Caricamento del dataset {DATASET_NAME}...")
dataset_dict = load_dataset(DATASET_NAME)

# Se lo split 'validation' non esiste, prendiamo parte dal 'train' split
if 'validation' not in dataset_dict:
    train_validation_split = dataset_dict['train'].train_test_split(test_size=0.2, seed=42)
    dataset_dict['train'] = train_validation_split['train']
    dataset_dict['validation'] = train_validation_split['test']

# Se il subset è specificato, selezionare un subset del train_dataset
if args.subset > 0:
    dataset_dict['train'] = dataset_dict['train'].shuffle(seed=42).select(range(args.subset))

# Estrazione dei tag NER unici dal dataset + "O"
print("Estrazione dei tag NER unici...")
unique_tags = set()
for example in dataset_dict['train']:
    for tag in example['ner_tags']:
        unique_tags.add(tag)

if 'O' in unique_tags:
    unique_tags.remove('O')
NER_TAGS = ['O'] + sorted(list(unique_tags))
# Utilità
TAG_TO_ID = {tag: i for i, tag in enumerate(NER_TAGS)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}

print(f"Tag NER trovati: {NER_TAGS}")

# Preparare il tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Funzione per tokenize e allineare le etichette
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

# Elaborare i dataset
processed_train_dataset = dataset_dict['train'].map(tokenize_and_align, batched=True, remove_columns=dataset_dict['train'].column_names)
processed_validation_dataset = dataset_dict['validation'].map(tokenize_and_align, batched=True, remove_columns=dataset_dict['validation'].column_names)
processed_test_dataset = dataset_dict['test'].map(tokenize_and_align, batched=True, remove_columns=dataset_dict['test'].column_names)

# Imposta il formato dei tensori di PyTorch
processed_train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
processed_validation_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
processed_test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
train_dataloader = DataLoader(processed_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(processed_validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(processed_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## TRAINING DEL MODELLO

# Inizializza il modello
model = NERPrefixTuningModel(
    model_name=MODEL_NAME,
    ner_tags=NER_TAGS,
    prefix_length=PREFIX_LENGTH,
    mid_dim=MID_DIM
).to(DEVICE)

# Inizializza ottimizzatore
trainable_params = list(model.prefix_module.parameters()) + list(model.classifier.parameters())
optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Liste per memorizzare le perdite di ogni epoca
all_train_losses = []
all_validation_losses = []

best_validation_loss = float('inf')
patience = 2 # Numero di epoche da aspettare prima di fermare l'addestramento
patience_counter = 0 

print("\nInizio dell'addestramento...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    # tqdm progress bar
    for batch in tqdm(train_dataloader, desc=f"Epoca {epoch+1}/{NUM_EPOCHS}"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Calcolo la loss e appiattire outputs e label
        outputs = outputs.view(-1, len(NER_TAGS))
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        
        # Backpropagation e aggioramento dei pesi
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_dataloader)

    # Memorizza la perdita di addestramento
    all_train_losses.append(avg_loss)

    # VALIDATION LOOP
    model.eval()
    validation_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Epoca {epoch+1}/{NUM_EPOCHS} Val"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Calcolo la loss e appiattire outputs e label
            outputs = outputs.view(-1, len(NER_TAGS))
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            
            validation_loss += loss.item()

            

    avg_validation_loss = validation_loss / len(validation_dataloader)

    # Memorizza la perdita di validazione
    all_validation_losses.append(avg_validation_loss)
    print(f"Epoca {epoch+1}/{NUM_EPOCHS}, Perdita (Loss): {avg_loss:.4f}, Perdita di Validazione (Loss): {avg_validation_loss:.4f}")

    # Salvo il modello solo se la loss della validazione è minore alla versione migliore
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        patience_counter = 0
        # Salvataggio del modello migliore
        torch.save(model.state_dict(), FILE_NAME)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping attivato. Nessun miglioramento per {patience} epoche. Addestramento interrotto.")
            break
    
print("\nAddestramento completato!")

# Stampa finale dei risultati
print("\nRisultati dell'addestramento per ogni epoca:")
for i in range(NUM_EPOCHS):
    print(f"Epoca {i+1}/{NUM_EPOCHS}, Perdita (Loss): {all_train_losses[i]:.4f}, Perdita di Validazione (Loss): {all_validation_losses[i]:.4f}")

## TEST DEL MODELLO

""" # Carica lo stato addestrato del modello
print("\nCaricamento del modello per la valutazione...")
model.load_state_dict(torch.load(FILE_NAME, map_location=DEVICE))
model.eval()

# Variabili per memorizzare le predizioni e le etichette reali
true_labels = []
predicted_labels = []

print("\nInizio della valutazione sul set di test...")
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Valutazione"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Le predizioni sono le classi con lo score più alto
        predictions = torch.argmax(outputs, dim=-1)

        # Allinea le predizioni e le etichette reali (ignora i token con -100)
        for i in range(labels.shape[0]):
            true_tags = []
            pred_tags = []
            
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    true_tags.append(ID_TO_TAG[labels[i, j].item()])
                    pred_tags.append(ID_TO_TAG[predictions[i, j].item()])
            
            true_labels.append(true_tags)
            predicted_labels.append(pred_tags)

print("\nReport di classificazione:")
print(classification_report(true_labels, predicted_labels, digits=4)) """