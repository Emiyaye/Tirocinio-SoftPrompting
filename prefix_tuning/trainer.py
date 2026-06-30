import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup 
from torch.optim import AdamW
import torch.nn.functional as F
from typing import List, Dict, Tuple
from soft_ner_model import NERSoftPromptModel
from prefix_tuning.prefix_ner_model import NERPrefixTuningModel
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
    
    # Inizializza ottimizzatore e criterio
    # prefix_module e classifier sono parametri addestrabili
    trainable_params = list(model.prefix_module.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # LinearLR
    num_training_steps = len(dataloaders['train']) * num_epochs
    num_warmup_steps = num_warmup_steps if num_warmup_steps > 0 else int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

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
        for batch in tqdm(dataloaders['train'], desc=f"Epoca {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Foward pass
            outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            
            # Appiattimento per CrossEntropy (Batch * SeqLen, NumTags)
            loss = criterion(outputs.view(-1, len(ner_tags)), batch['labels'].to(device).view(-1))
            
            # BackPropagation
            loss.backward()
            
            # Aggiornamento dei pesi e del learning rate
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
        # TODO: Remove
        #print(f"Epoca {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        # Logica di Early Stopping: se la loss di validazione scende, salviamo il modello
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
    """Funzione separata per il testing e la valutazione finale."""
    dataloaders, ner_tags, _ = prepare_data(dataset_name, model_name, max_seq_len, -1, batch_size, device)
    id_to_tag = {i: tag for i, tag in enumerate(ner_tags)}
    
    if "roberta-base-uncased" in model_name.lower():
        model_name = "roberta-base"
    model = NERSoftPromptModel(model_name=model_name, ner_tags=ner_tags, prefix_length=prefix_length, mid_dim=mid_dim).to(device)
    
    if os.path.exists(file_name):
        model.load_state_dict(torch.load(file_name, map_location=device, weights_only=True))
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

def interpret_soft_tokens(model, tokenizer, file_name, k=10, bio_lexicon=None):
    model.eval()
    with torch.no_grad():
        # Estrazione degli embedding dei soft token
        soft_embeddings = model.prefix_module(bsz=1).squeeze(0)
        
        # Salva il tensore degli embeddings per analisi future
        tensor_save_path = file_name.replace(".pth", "_prefix_embeds.pt")
        torch.save(soft_embeddings.cpu(), tensor_save_path)
        
        # Recupero della matrice dei pesi del vocabolario del backbone (V x H)
        word_embeddings = model.encoder.get_input_embeddings().weight
        
        # Normalizzazione per il calcolo della Cosine Similarity
        soft_norm = F.normalize(soft_embeddings, dim=-1)
        word_norm = F.normalize(word_embeddings, dim=-1)
        
        # Cosine similarity (L x V)
        scores_matrix = torch.matmul(soft_norm, word_norm.T)
    
    vocab_size = word_embeddings.size(0)
    valid_vocab_indices = []
    valid_tokens = []
    punct_set = set(string.punctuation)
    
    # 5. Filtro del vocabolario per rimuovere i token rumorosi
    for idx in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        
        # Skip token del Transformer
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token, tokenizer.pad_token, tokenizer.unk_token]:
            continue
        if token.startswith("[unused") or token.startswith("<") or token.endswith(">"):
            continue
            
        # SKip  subword dei modelli (BERT, RoBERTa..)
        if token.startswith("##") or token.startswith("Ġ") or token.startswith("â"):
            continue
            
        # Skip token di punteggiatura
        if len(token) <= 1 and token in punct_set:
            continue
        if all(c in punct_set for c in token):
            continue
            
        valid_vocab_indices.append(idx)
        valid_tokens.append(token)
        
    valid_vocab_indices = torch.tensor(valid_vocab_indices, device=scores_matrix.device)
    
    # (L x V_filtrato)
    filtered_scores = scores_matrix[:, valid_vocab_indices]
    
    # Estrazione dei top-k punteggi e indici locali relativi al vocabolario filtrato
    top_scores, top_filtered_indices = torch.topk(filtered_scores, k=k, dim=-1)
    
    print()
    print(f"NEAREST-NEIGHBOR ANALYSIS(Top-{k})")
    print(f"Contrassegno: * = token identificato come dominio biomedicale")
    print(f"Tensore dei prompt salvato in: {tensor_save_path}\n")
    
    total_bio_count = 0
    total_tokens_evaluated = soft_embeddings.size(0)
    all_top_scores = []
    
    for i in range(total_tokens_evaluated):
        neighbor_strings = []
        bio_matches_in_top_k = 0
        
        for j in range(k):
            local_idx = top_filtered_indices[i, j].item()
            score = top_scores[i, j].item()
            token_text = valid_tokens[local_idx]
            all_top_scores.append(score)
            
            # Controllo dell'appartenenza al dominio (Domain Alignment)
            is_bio = False
            if bio_lexicon is not None:
                if token_text.lower() in bio_lexicon:
                    is_bio = True
            else:
                # Euristica basata su pattern medici comuni
                bio_patterns = ('tion', 'gical', 'itis', 'path', 'acid', 'gene', 'protein', 
                                'cell', 'mab', 'ase', 'cine', 'vir', 'num', 'tox', 'in', 'med')
                
                if any(ext in token_text.lower() for ext in bio_patterns) or (len(token_text) > 3 and any(c.isdigit() for c in token_text)):
                    is_bio = True
            
            if is_bio:
                bio_matches_in_top_k += 1
                token_display = f"{token_text}*( {score:.2f} )"
            else:
                token_display = f"{token_text}( {score:.2f} )"
                
            neighbor_strings.append(token_display)
            
        bio_percentage_per_token = (bio_matches_in_top_k / k) * 100
        total_bio_count += bio_matches_in_top_k
        
        print(f"Virtual Token {i+1:02d} [Bio: {bio_percentage_per_token:5.1f}%]: {' | '.join(neighbor_strings)}")
        
    global_bio_ratio = (total_bio_count / (total_tokens_evaluated * k)) * 100
    mean_top_cosine = np.mean(all_top_scores)
    print("\n\n")
    print(f"DOMAIN ALIGNMENT SCORE (Percentuale Bio): {global_bio_ratio:.2f}%")
    print(f"COSINE SIMILARITY MEDIA (Top-{k}):         {mean_top_cosine:.4f}")
    print("\n\n")
    return global_bio_ratio, mean_top_cosine