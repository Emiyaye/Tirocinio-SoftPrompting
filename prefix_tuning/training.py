import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup 
from torch.optim import AdamW
from typing import List, Dict, Tuple
from ner_prefix_tuning_model import NERPrefixTuningModel
from datasets import load_dataset
from tqdm import tqdm
from seqeval.metrics import classification_report
import argparse
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def train_ner_prefix_tuning_model(
    model_name: str,
    dataset_name: str,
    prefix_length: int = 10,
    mid_dim: int = 512,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 5,
    max_seq_len: int = 128,
    subset_size: int = -1,
    file_name: str = "_.pth",
    patience: int = 2,
    num_warmup_steps: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[List[float], List[float], str]:
    
    # Carica il dataset da Hugging Face
    eprint(f"Caricamento del dataset {dataset_name} su {device}...")
    dataset_dict = load_dataset(dataset_name)

    # Se lo split 'validation' non esiste, prendiamo parte dal 'train' split
    if 'validation' not in dataset_dict:
        train_validation_split = dataset_dict['train'].train_test_split(test_size=0.2, seed=42)
        dataset_dict['train'] = train_validation_split['train']
        dataset_dict['validation'] = train_validation_split['test']

    # Se il subset è specificato, selezionare un subset del train_dataset
    if subset_size > 0:
        dataset_dict['train'] = dataset_dict['train'].shuffle(seed=42).select(range(subset_size))
        dataset_dict['validation'] = dataset_dict['validation'].shuffle(seed=42).select(range(int(subset_size/10)))


    # Estrazione dei tag NER unici dal dataset + "O"
    eprint("Estrazione dei tag NER unici...")
    unique_tags = set()
    for example in dataset_dict['train']:
        for tag in example['ner_tags']:
            unique_tags.add(tag)

    if 'O' in unique_tags:
        unique_tags.remove('O')
    ner_tags = ['O'] + sorted(list(unique_tags))
    # Utilità
    tag_to_id = {tag: i for i, tag in enumerate(ner_tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    eprint(f"Tag NER trovati: {ner_tags}")

    # Preparare il tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Funzione per tokenize e allineare le etichette
    def tokenize_and_align(examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        tokenized_inputs = tokenizer(
            examples['tokens'],
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
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
                label_ids[word_ids.index(word_idx)] = tag_to_id[current_tag]
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
    train_dataloader = DataLoader(processed_train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(processed_validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(processed_test_dataset, batch_size=batch_size, shuffle=False)

    ## TRAINING DEL MODELLO

    # Inizializza il modello
    model = NERPrefixTuningModel(
        model_name=model_name,
        ner_tags=ner_tags,
        prefix_length=prefix_length,
        mid_dim=mid_dim
    ).to(device=device)

    # Inizializza ottimizzatore e criterio
    # prefix_module e classifier sono parametri addestrabili
    trainable_params = list(model.prefix_module.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01, eps=1e-8)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # LinearLR
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_warmup_steps if num_warmup_steps > 0 else int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # CosineAnnealingLR
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    eprint(f"Configurazione Scheduler: Passi totali: {num_training_steps}, Passi Warmup: {num_warmup_steps}")

    ## TRAINING LOOP
    # Liste per memorizzare le perdite di ogni epoca
    all_train_losses = []
    all_validation_losses = []

    best_validation_loss = float('inf')
    patience_counter = 0 

    eprint("\nInizio dell'addestramento...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # tqdm progress bar
        for batch in tqdm(train_dataloader, desc=f"Epoca {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Calcolo la loss e appiattire outputs e label
            outputs = outputs.view(-1, len(ner_tags))
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            
            # Backpropagation e aggioramento dei pesi
            loss.backward()
            optimizer.step()
            # Aggiorna il learning rate dopo ogni epoca
            scheduler.step() 
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)

        # Memorizza la perdita di addestramento
        all_train_losses.append(avg_loss)

        # VALIDATION LOOP
        model.eval()
        validation_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(validation_dataloader, desc=f"Epoca {epoch+1}/{num_epochs} Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Calcolo la loss e appiattire outputs e label
                outputs = outputs.view(-1, len(ner_tags))
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                
                validation_loss += loss.item()

                

        avg_validation_loss = validation_loss / len(validation_dataloader)

        # Memorizza la perdita di validazione
        all_validation_losses.append(avg_validation_loss)
        eprint(f"Epoca {epoch+1}/{num_epochs}, Perdita (Loss): {avg_loss:.4f}, Perdita di Validazione (Loss): {avg_validation_loss:.4f}")

        # Salvo il modello solo se la loss della validazione è minore alla versione migliore
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            patience_counter = 0
            # Salvataggio del modello migliore
            torch.save(model.state_dict(), file_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                eprint(f"Early stopping attivato. Nessun miglioramento per {patience} epoche. Addestramento interrotto.")
                break
        
    eprint("\nAddestramento completato!")

    ## TEST DEL MODELLO
    if os.path.exists(file_name):
        eprint("\nCaricamento del modello migliore per la valutazione...")
        model.load_state_dict(torch.load(file_name, map_location=device))
        model.eval()
        
        true_labels = []
        predicted_labels = []

        eprint("\nInizio della valutazione sul set di test...")
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Valutazione"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                
                predictions = torch.argmax(outputs, dim=-1)

                # Allinea le predizioni e le etichette reali (ignora i token con -100)
                for i in range(labels.shape[0]):
                    true_tags = []
                    pred_tags = []
                    
                    for j in range(labels.shape[1]):
                        if labels[i, j] != -100:
                            true_tags.append(id_to_tag[labels[i, j].item()])
                            pred_tags.append(id_to_tag[predictions[i, j].item()])
                    
                    true_labels.append(true_tags)
                    predicted_labels.append(pred_tags)

        print("\nReport di classificazione finale:")
        print(classification_report(true_labels, predicted_labels, digits=4))

    return all_train_losses, all_validation_losses, file_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Addestra e valuta un modello NER con Prefix Tuning.')
    
    # Argomenti del modello e del dataset
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='Nome del modello base da Hugging Face.')
    parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr',
                        help='Nome del dataset da Hugging Face nel formato "user/dataset".')
    
    # Argomenti di Prefix Tuning
    parser.add_argument('--prefix_length', type=int, default=10,
                        help='Lunghezza del token (k) per il prefix tuning.')
    parser.add_argument('--mid_dim', type=int, default=512,
                        help='Dimensione intermedia (mid_dim) per la proiezione del prefisso.')

    # Argomenti di Addestramento
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Dimensione del batch per l\'addestramento e la valutazione.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Tasso di apprendimento (Learning Rate).')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Numero totale di epoche di addestramento.')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Lunghezza massima della sequenza.')
    parser.add_argument('--patience', type=int, default=4,
                        help='Numero di epoche senza miglioramento prima dell\'early stopping.')
    parser.add_argument('--subset_size', type=int, default=-1,
                        help='Dimensione del subset di dati da usare per l\'addestramento e il test (-1 per l\'intero dataset).')
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help='Numero di warp up per ogni step')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Dispositivo da utilizzare (es. "cuda", "cpu").')

    args = parser.parse_args()


    MODEL_NAME = args.model
    DATASET_NAME = args.dataset
    PREFIX_LENGTH = args.prefix_length
    MID_DIM = args.mid_dim
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    MAX_SEQ_LEN = args.max_seq_len
    SUBSET_SIZE = args.subset_size
    PATIENCE = args.patience
    NUM_WARMUP_STEPS = args.num_warmup_steps
    DEVICE = torch.device(args.device)

    # Nome file per il salvataggio
    file_name = MODEL_NAME.replace("/", "-") + "_" + DATASET_NAME.replace("/", "-") + "_GELU_token_lenght-" + str(PREFIX_LENGTH) + ".pth"

    print(f"model = {MODEL_NAME} dataset = {DATASET_NAME} token_lenght = {PREFIX_LENGTH}")
    
    
    train_losses, val_losses, _saved_model_path = train_ner_prefix_tuning_model(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        prefix_length=PREFIX_LENGTH,
        mid_dim=MID_DIM,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        max_seq_len=MAX_SEQ_LEN,
        subset_size=SUBSET_SIZE,
        file_name=file_name,
        patience=PATIENCE,
        num_warmup_steps=NUM_WARMUP_STEPS,
        device=DEVICE
    )
    for i in range(len(val_losses)):
        print(f"Epoca {i+1}, Perdita (Loss): {train_losses[i]:.4f}, Perdita di Validazione (Loss): {val_losses[i]:.4f}")
    print()

