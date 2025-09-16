#!/usr/bin/env python3

import datetime
from functools import reduce
import operator
import os
import random
import sys

from typing import List, Optional, Sequence, Tuple, Type, Union

#try:
from typing import Literal
#except ImportError:
    #from typing_extensions import Literal

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import tqdm
from datasets import load_dataset

from utils import CollateForTraining



# Classe per gestire i soft prompts multipli
class MultiPrefix(nn.Module):
    """
    Gestisce i soft prompts per ogni tag NER.
    """
    def __init__(self, num_tags, prefix_size, embedding_size, n_layers):
        """
        Args:
            num_tags: Numero di tag NER.
            prefix_size: La lunghezza di ogni soft prompt.
            embedding_size: La dimensione del vettore di embedding di ogni token.
            n_layers: Il numero di strati dell'encoder del modello.
        """
        super().__init__()
        #                Negative Tag
        # Shape: (num_tags + 1, n_layers, prefix_size, embedding_size)
        self.prefixes = nn.Parameter(
            torch.normal(0, 0.02, size=(num_tags + 1, n_layers, prefix_size, embedding_size))
        )
        self.num_tags = num_tags
        self.prefix_size = prefix_size

    def get_prefix_by_tag(self, tag_id):
        return self.prefixes[tag_id, 0]

def prefix_ner_tuning(
    model_name: Literal["dslim/bert-base-NER", "t5"],
    dataset: str,
    #prefix_mode: Literal["Classic"] = "Classic",
    prefix_size: int = 20, # len del soft prompt (numero di token virtuali)
    max_seq_len: int = 128, # max len sequenza di input
    max_batch_size: int = 8,
    lr: float = 1e-2, # learning rate
    epochs: int = 10,
    seed: int = 42,
    train_subset: Optional[int] = None, # subset del dataset di training usato, se specificato
    test_subset: Optional[int] = None, # subset di dati di test usato, se specificato
    short_test: Optional[int] = None, # test
):
    # Set up
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    print(f"Seed is set to {seed}")

    ## LOAD MODEL
    
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_layers = model.config.num_hidden_layers
    embedding_size = model.config.hidden_size
    
    # congelare i parametri del modello
    for name, param in model.named_parameters():
        if 'classifier' not in name: # eccetto l'ultimo strato
            param.requires_grad = False
    
    # sostituzione layer finale
    model.classifier = nn.Identity()
    
    model.cuda()
    

    ## PREPARE DATASET
    try:
        # datasets version = 3.6.0
        ds_train = load_dataset(dataset, trust_remote_code=True, split="train")
        ds_test = load_dataset(dataset, trust_remote_code=True, split="test")

    except Exception as e:
        raise RuntimeError(f"Errore durante il caricamento del dataset '{dataset}': {e}")
    
    # get tag dal dataset
    """ label_feature_name = None
    for feature_name in ds_train.features:
        if "label" in feature_name.lower() or "tag" in feature_name.lower():
            label_feature_name = feature_name
            break

    if label_feature_name is None:
        raise ValueError("Non è stato possibile trovare una feature contenente le etichette nel dataset.") """
    label_feature_name="ner_tags"
    # Ottieni la mappa dei tag usando il nome della feature trovato
    label_map = ds_train.features[label_feature_name].feature.names
    id2label = {i: label for i, label in enumerate(label_map)}
    num_tags = len(label_map)
    
    # tag per la classe negativa
    id2label[num_tags] = 'NEGATIVE' 
    # print(f"Numero di tag: {num_tags}, Mappa dei tag: {label_map}")
    
    if train_subset is not None and len(ds_train) > train_subset:
        ds_train = Subset(ds_train, random.sample(range(len(ds_train)), train_subset))
    if test_subset is not None and len(ds_test) > test_subset:
        ds_test = Subset(ds_test, random.sample(range(len(ds_test)), test_subset))
    print(
        f"Dataset loaded: {len(ds_train)} training samples and {len(ds_test)} test samples."
    )
    
    collate_fn = CollateForTraining(
    tokenizer=tokenizer,
    max_seq_len=max_seq_len-prefix_size
)

    dl_train = DataLoader(
        ds_train, batch_size=max_batch_size, shuffle=True, collate_fn=collate_fn
    )
    dl_test = DataLoader(
        ds_test, batch_size=max_batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    ## SETUP MULTIPLE PREFIX
    multi_prefix_module = MultiPrefix(
        num_tags=num_tags, 
        prefix_size=prefix_size, 
        embedding_size=embedding_size, 
        n_layers=n_layers
    ).cuda()
    
    # placeholder per la rappresentazione testuale del prefisso per i token
    prefix_str = " ".join(["P"] * prefix_size)
    prefix_tokens = torch.tensor(tokenizer.encode(prefix_str, add_special_tokens=False), dtype=torch.long)
    assert prefix_size == len(prefix_tokens)

    # layer finale di classificazione
    # che prenderà l'embedding di ogni token e lo proietterà a uno score
    class FinalClassifier(nn.Module):
        def __init__(self, embedding_size):
            super().__init__()
            self.linear = nn.Linear(embedding_size, 1)

        def forward(self, x):
            return self.linear(x)

    final_classifier = FinalClassifier(embedding_size).cuda()
    ## SETUP THE OPTIMIZER
    optimizer = torch.optim.Adam(list(multi_prefix_module.parameters()) + list(final_classifier.parameters()), lr=lr)
    
    ## LOOP DI TRAINING
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm.tqdm(
            enumerate(dl_train),
            desc=f"Training epoch {epoch+1}/{epochs}",
            total=len(dl_train),
        )
        for b_idx, b in pbar:
            bsz = b.input_ids.size(0)
            input_ids, attention_mask, labels = (
            b['input_ids'].cuda(),
            b['attention_mask'].cuda(),
            b['labels'].cuda(),
        )

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                tag_scores = []
                for tag_id in range(num_tags + 1):
                    
                    current_prefix_embeds = multi_prefix_module.get_prefix_by_tag(tag_id)
                    current_prefix_embeds = current_prefix_embeds.unsqueeze(0).repeat(bsz, 1, 1)

                    # embedding dei token originali dal modello base
                    model_core = get_model_core(model)
                    input_embeds = model_core.embeddings(input_ids)

                    # merge prefissi con gli embedding dei token originali
                    combined_embeds = torch.cat(
                        [current_prefix_embeds, input_embeds], dim=1
                    )

                    prefixed_attention_mask = torch.cat(
                        [
                            torch.ones(bsz, prefix_size, device=attention_mask.device),
                            attention_mask,
                        ],
                        dim=1,
                    )

                    # forward pass del modello BERT usando gli embedding e la maschera combinata
                    outputs = model_core(
                        inputs_embeds=combined_embeds,
                        attention_mask=prefixed_attention_mask,
                        output_hidden_states=True
                    )
                    
                    token_embeddings = outputs.hidden_states[-1][:, prefix_size:]
                    score = final_classifier(token_embeddings)
                    tag_scores.append(score)

                # Unisci gli score e calcola la loss
                tag_scores = torch.cat(tag_scores, dim=-1)
                loss = nn.functional.cross_entropy(
                    tag_scores.transpose(1, 2),
                    labels,
                    ignore_index=-100
                )
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.4e}"})

    ## LOOP DI TEST
    model.eval()
    all_predictions = []
    all_true_labels = []

    pbar = tqdm.tqdm(enumerate(dl_test),desc=f"Testing", total=len(dl_test))
    for b_idx, b in pbar:
        bsz = b['input_ids'].size(0)
        input_ids, attention_mask, labels = (
            b['input_ids'].cuda(),
            b['attention_mask'].cuda(),
            b['labels'].cuda(),
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            tag_scores = []
            for tag_id in range(num_tags + 1):
                current_prefix_embeds = multi_prefix_module.get_prefix_by_tag(tag_id)
                current_prefix_embeds = current_prefix_embeds.unsqueeze(0).repeat(bsz, 1, 1)

                model_core = get_model_core(model)
                input_embeds = model_core.embeddings(input_ids)

                combined_embeds = torch.cat(
                    [current_prefix_embeds, input_embeds], dim=1
                )

                prefixed_attention_mask = torch.cat(
                    [
                        torch.ones(bsz, prefix_size, device=attention_mask.device),
                        attention_mask,
                    ],
                    dim=1,
                )

                outputs = model_core(
                    inputs_embeds=combined_embeds,
                    attention_mask=prefixed_attention_mask,
                    output_hidden_states=True # per hidden states
                )
                
                token_embeddings = outputs.hidden_states[-1][:, prefix_size:]
                score = final_classifier(token_embeddings)
                tag_scores.append(score)

            tag_scores = torch.cat(tag_scores, dim=-1)
            predictions = torch.argmax(tag_scores, dim=-1)
        
        all_predictions.extend(predictions.cpu().tolist())
        all_true_labels.extend(labels.cpu().tolist())
        
    
    # salva i soft prompts addestrati
    dirname = f"trained_prefixes"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    safe_model_name = model_name.replace('/', '-')
    filename = f"{dirname}/{safe_model_name}_{dataset}_ner_{prefix_size}_{timestamp}_{epoch}.pt"
    
    to_store = {f"prefix_{id2label[i]}": multi_prefix_module.get_prefix_by_tag(i) for i in range(num_tags + 1)}
    to_store["final_classifier_state"] = final_classifier.state_dict()
    torch.save(to_store, filename)

# util
def get_model_core(model):
    core_names = ["bert", "t5"]
    for name in core_names:
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("Modello non trovato.")

if __name__ == "__main__":
    #fire.Fire(prefix_ner_tuning)
    #dmis-lab/biobert-v1.1
    #dslim/bert-base-NER

    prefix_ner_tuning(model_name="dslim/bert-base-NER", dataset="conll2003", epochs=1, train_subset=10, test_subset=10)