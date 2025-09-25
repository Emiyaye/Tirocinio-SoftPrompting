import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List
from prefix_ner_module import NERPrefixModule
    
class NERPrefixTuningModel(nn.Module):
    def __init__(self, model_name: str, ner_tags: List[str], prefix_length: int = 10, mid_dim: int = 512):
        super().__init__()
        
        # Carica l'encoder e congelare i parametri
        self.encoder = AutoModel.from_pretrained(model_name) #!! DistilBert, non ha un argomento past_key_values
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        encoder_config = self.encoder.config
        self.ner_tags = ner_tags
        self.num_tags = len(ner_tags)

        # Inizializza un singolo modulo di prefix tuning che genera Key e Value per tutti i layer
        self.prefix_module = NERPrefixModule(
            encoder_config=encoder_config,
            prefix_length=prefix_length,
            mid_dim=mid_dim
        )
        self.preseqlen = prefix_length
        
        # Un layer di classificazione finale con un numero di output pari al numero di tag
        self.classifier = nn.Linear(encoder_config.hidden_size, self.num_tags)
        # Per gestire diversi attributi di dropout (es. BERT vs DistilBERT)
        dropout_prob = getattr(encoder_config, 'hidden_dropout_prob', getattr(encoder_config, 'dropout', 0.1))
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        # Genera Key e Value
        past_key_values = self.prefix_module(bsz=batch_size) 
        
        prefix_attention_mask = torch.ones(batch_size, self.preseqlen, device=input_ids.device)
        updated_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        # Forward pass encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=updated_attention_mask,
            past_key_values=past_key_values #!! DistilBert, non ha un argomento past_key_values
        )
        
        last_hidden_states = outputs.last_hidden_state
        
        # (bsz, seq_len, n_embd)
        text_hidden_states = last_hidden_states 
        
        # Dropout e layer lineare
        tag_scores = self.classifier(self.dropout(text_hidden_states))
        
        return tag_scores
    