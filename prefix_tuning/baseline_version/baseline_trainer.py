import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from torch.optim import AdamW
import torch.nn.functional as F
from typing import List, Dict, Tuple
from soft_ner_model import NERSoftPromptModel
from prefix_ner_modul import NERPrefixTuningModel
from datasets import load_dataset
from tqdm import tqdm
from seqeval.metrics import classification_report
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def prepare_data(dataset_name, model_name, max_seq_len, subset_size, batch_size, device):
    """Funzione di utilità per caricare e processare i dati."""
    
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
        dataset_dict['validation'] = dataset_dict['validation'].shuffle(seed=42).select(range(max(1, int(subset_size/10))))

    # Estrazione dei tag NER unici dal dataset + "O"
    unique_tags = set()
    for example in dataset_dict['train']:
        for tag in example['ner_tags']:
            unique_tags.add(tag)
    if 'O' in unique_tags: unique_tags.remove('O')
    ner_tags = ['O'] + sorted(list(unique_tags))
    tag_to_id = {tag: i for i, tag in enumerate(ner_tags)}

    is_roberta = "roberta" in model_name.lower()
    
    # Per roberta base uncased
    def get_tokenizer(model_name):
        if "roberta-base-uncased" in model_name.lower():
            target_model = "roberta-base"
            do_lower = True
        else:
            target_model = model_name
            do_lower = False

        tokenizer = AutoTokenizer.from_pretrained(
            target_model, 
            add_prefix_space=True if is_roberta else False,
            do_lower_case=do_lower
        )
        return tokenizer
    
    tokenizer = get_tokenizer(model_name)

    # Funzione per tokenize e allineare le etichette
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(examples['tokens'], padding="max_length", truncation=True, 
                                     max_length=max_seq_len, is_split_into_words=True)
        labels = []
        for i, label_list in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100] * len(word_ids)
            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx: continue
                label_ids[idx] = tag_to_id[label_list[word_idx]]
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_ds = dataset_dict.map(tokenize_and_align, batched=True, remove_columns=dataset_dict['train'].column_names)
    processed_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    dataloaders = {
        'train': DataLoader(processed_ds['train'], batch_size=batch_size, num_workers=2, shuffle=True),
        'validation': DataLoader(processed_ds['validation'], batch_size=batch_size, num_workers=2),
        'test': DataLoader(processed_ds['test'], batch_size=batch_size, num_workers=2)
    }
    
    return dataloaders, ner_tags, tokenizer


class NERBaselineFineTuningModel(nn.Module):
    def __init__(self, model_name: str, ner_tags: list):
        super().__init__()
        # Carica la stessa identica backbone, lasciando tutti i pesi sbloccati
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Classificatore lineare identico alla versione Prefix/Soft Prompt
        self.classifier = nn.Linear(self.hidden_size, len(ner_tags))

    def forward(self, input_ids, attention_mask):
        # Passaggio diretto all'encoder senza alterazione o aggiunta di prefissi virtuali
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Consideriamo lo spazio degli hidden state della sequenza reale
        text_hidden = out.last_hidden_state
        
        logits = self.classifier(text_hidden)
        return logits


def train_ner_baseline_model(
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    num_epochs: int = 5,
    max_seq_len: int = 128,
    subset_size: int = -1,
    file_name: str = "model_baseline.pth",
    patience: int = 2,
    num_warmup_steps: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[List[float], List[float], str]:
    
    dataloaders, ner_tags, _ = prepare_data(dataset_name, model_name, max_seq_len, subset_size, batch_size, device)
    
    if "roberta-base-uncased" in model_name.lower():
        model_name = "roberta-base"
        
    model = NERBaselineFineTuningModel(model_name=model_name, ner_tags=ner_tags).to(device)
    
    # Rendiamo esplicitamente tutti i parametri del modello soggetti ad aggiornamento del gradiente
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_training_steps = len(dataloaders['train']) * num_epochs
    num_warmup_steps = num_warmup_steps if num_warmup_steps > 0 else int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    eprint(f"Configurazione Scheduler Baseline: Passi totali: {num_training_steps}, Passi Warmup: {num_warmup_steps}")
    
    all_train_losses = []
    all_validation_losses = []
    best_validation_loss = float('inf')
    patience_counter = 0 

    eprint("\nInizio dell'addestramento Baseline (Full Fine-Tuning)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoca Baseline {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Forward pass diretto
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = criterion(outputs.view(-1, len(ner_tags)), batch['labels'].to(device).view(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(dataloaders['train'])
        all_train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloaders['validation'], desc=f"Epoca Baseline {epoch+1}/{num_epochs} Val"):
                outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                v_loss += criterion(outputs.view(-1, len(ner_tags)), batch['labels'].to(device).view(-1)).item()
        
        avg_val_loss = v_loss / len(dataloaders['validation'])
        all_validation_losses.append(avg_val_loss)
        eprint(f"Epoca {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        print(f"Epoca {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), file_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                eprint(f"Early stopping attivato (Baseline). Nessun miglioramento per {patience} epoche. Addestramento interrotto.")
                break
                
    return all_train_losses, all_validation_losses, file_name

def evaluate_baseline_model(model_name, dataset_name, file_name, max_seq_len, batch_size, device):
    """Funzione di valutazione finale per il modello baseline."""
    dataloaders, ner_tags, _ = prepare_data(dataset_name, model_name, max_seq_len, -1, batch_size, device)
    id_to_tag = {i: tag for i, tag in enumerate(ner_tags)}
    
    if "roberta-base-uncased" in model_name.lower():
        model_name = "roberta-base"
    model = NERBaselineFineTuningModel(model_name=model_name, ner_tags=ner_tags).to(device)
    
    if os.path.exists(file_name):
        model.load_state_dict(torch.load(file_name, map_location=device))
        eprint(f"Modello baseline caricato da {file_name}")
    
    model.eval()
    true_labels = []
    predicted_labels = []

    eprint("\nInizio della valutazione sul set di test (Baseline)...")
    with torch.no_grad():
        for batch in tqdm(dataloaders['test'], desc="Valutazione Baseline"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=batch['attention_mask'].to(device))
            predictions = torch.argmax(outputs, dim=-1)

            for i in range(labels.shape[0]):
                t_tags = []
                p_tags = []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        t_tags.append(id_to_tag[labels[i, j].item()])
                        p_tags.append(id_to_tag[predictions[i, j].item()])
                true_labels.append(t_tags)
                predicted_labels.append(p_tags)

    report = classification_report(true_labels, predicted_labels, digits=4)
    print("\nReport di classificazione finale Baseline:")
    print(report)
    return report

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
    file_name: str = "model.pth",
    patience: int = 2,
    num_warmup_steps: int = 0,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[List[float], List[float], str]:
    
    dataloaders, ner_tags, _ = prepare_data(dataset_name, model_name, max_seq_len, subset_size, batch_size, device)
    
    if "roberta-base-uncased" in model_name.lower():
        model_name = "roberta-base"
    model = NERSoftPromptModel(model_name=model_name, ner_tags=ner_tags, prefix_length=prefix_length, mid_dim=mid_dim).to(device)
    
    trainable_params = list(model.prefix_module.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_training_steps = len(dataloaders['train']) * num_epochs
    num_warmup_steps = num_warmup_steps if num_warmup_steps > 0 else int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    eprint(f"Configurazione Scheduler: Passi totali: {num_training_steps}, Passi Warmup: {num_warmup_steps}")
    
    all_train_losses = []
    all_validation_losses = []
    best_validation_loss = float('inf')
    patience_counter = 0 

    eprint("\nInizio dell'addestramento...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoca {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            loss = criterion(outputs.view(-1, len(ner_tags)), batch['labels'].to(device).view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(dataloaders['train'])
        all_train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloaders['validation'], desc=f"Epoca {epoch+1}/{num_epochs} Val"):
                outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                v_loss += criterion(outputs.view(-1, len(ner_tags)), batch['labels'].to(device).view(-1)).item()
        
        avg_val_loss = v_loss / len(dataloaders['validation'])
        all_validation_losses.append(avg_val_loss)
        eprint(f"Epoca {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), file_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                eprint(f"Early stopping attivato. Nessun miglioramento per {patience} epoche. Addestramento interrotto.")
                break
                
    return all_train_losses, all_validation_losses, file_name


def evaluate_model(model_name, dataset_name, file_name, prefix_length, mid_dim, max_seq_len, batch_size, device):
    dataloaders, ner_tags, _ = prepare_data(dataset_name, model_name, max_seq_len, -1, batch_size, device)
    id_to_tag = {i: tag for i, tag in enumerate(ner_tags)}
    
    if "roberta-base-uncased" in model_name.lower():
        model_name = "roberta-base"
    model = NERSoftPromptModel(model_name=model_name, ner_tags=ner_tags, prefix_length=prefix_length, mid_dim=mid_dim).to(device)
    
    if os.path.exists(file_name):
        model.load_state_dict(torch.load(file_name, map_location=device))
        eprint(f"Modello caricato da {file_name}")
    
    model.eval()
    true_labels = []
    predicted_labels = []

    eprint("\nInizio della valutazione sul set di test...")
    with torch.no_grad():
        for batch in tqdm(dataloaders['test'], desc="Valutazione"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=batch['attention_mask'].to(device))
            predictions = torch.argmax(outputs, dim=-1)

            for i in range(labels.shape[0]):
                t_tags = []
                p_tags = []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        t_tags.append(id_to_tag[labels[i, j].item()])
                        p_tags.append(id_to_tag[predictions[i, j].item()])
                true_labels.append(t_tags)
                predicted_labels.append(p_tags)

    report = classification_report(true_labels, predicted_labels, digits=4)
    print("\nReport di classificazione finale:")
    print(report)
    return report


def interpret_soft_tokens(model, tokenizer, file_name, k=5):
    model.eval()
    with torch.no_grad():
        soft_embeddings = model.prefix_module(bsz=1).squeeze(0)
        tensor_save_path = file_name.replace(".pth", "_prefix_embeds.pt")
        torch.save(soft_embeddings.cpu(), tensor_save_path)
        
        word_embeddings = model.encoder.get_input_embeddings().weight
        soft_norm = F.normalize(soft_embeddings, dim=-1)
        word_norm = F.normalize(word_embeddings, dim=-1)
        
        scores = torch.matmul(soft_norm, word_norm.T)
        top_tokens, top_indices = torch.topk(scores, k=k, dim=-1)
        
        print(f"\nAnalisi Soft Prompt (Salvati in: {tensor_save_path})")
        for i in range(soft_embeddings.size(0)):
            tokens = tokenizer.convert_ids_to_tokens(top_indices[i])
            print(f"Virtual Token {i+1:02d}: {' | '.join(tokens)}")