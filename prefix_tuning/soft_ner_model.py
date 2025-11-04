import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List
from soft_ner_module import NERSoftModule
    
class NERSoftPromptModel(nn.Module):
    def __init__(self, model_name: str, ner_tags: List[str], prefix_length: int = 10, mid_dim: int = 512):
        super().__init__()
        
        # Carica l'encoder e congelare i parametri
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        encoder_config = self.encoder.config
        self.ner_tags = ner_tags
        self.num_tags = len(ner_tags)
        
        """ # Un modulo di prefix tuning per ogni tag NER (tag negativo aggiunto precedentemente)
        self.tag_prompts = nn.ModuleDict()
        for tag in self.ner_tags:
            self.tag_prompts[tag] = NERSoftModule(
                encoder_config=encoder_config,
                prefix_length=prefix_length,
                mid_dim=mid_dim
            )  """
        # Inizializza un singolo modulo di prefix tuning
        self.prefix_module = NERSoftModule(
            encoder_config=encoder_config,
            prefix_length=prefix_length,
            mid_dim=mid_dim
        )
            
        # Un layer di classificazione finale
        self.classifier = nn.Linear(encoder_config.hidden_size, self.num_tags)
        # Per gestire diversi attributi di dropout (es. BERT vs DistilBERT)
        dropout_prob = getattr(encoder_config, 'hidden_dropout_prob', getattr(encoder_config, 'dropout', 0.1))
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        prefix_embeddings = self.prefix_module(bsz=batch_size)
        input_embeddings = self.encoder.get_input_embeddings()(input_ids)
        
        # Soft prompt + input embedding
        encoder_input_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)
        
        # Update della maschera di attenzione
        prefix_attention_mask = torch.ones(batch_size, prefix_embeddings.shape[1], device=input_ids.device)
        updated_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        # Forward pass encoder
        encoder_output = self.encoder(
            attention_mask=updated_attention_mask,
            inputs_embeds=encoder_input_embeddings
        )
        
        last_hidden_states = encoder_output.last_hidden_state
        text_hidden_states = last_hidden_states[:, prefix_embeddings.shape[1]:]
        
        # Dropout e layer lineare
        tag_scores = self.classifier(self.dropout(text_hidden_states))
        
        return tag_scores
    
    """ 
        
        def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        # Matrice per salvare gli score per ogni tag
        scores_matrix = torch.zeros(batch_size, seq_len, self.num_tags, device=input_ids.device)

        input_embeddings = self.encoder.get_input_embeddings()(input_ids)
        
        # Per ogni tag per eseguire un forward pass separato
        for i, tag in enumerate(self.ner_tags):
            
            # Soft prompt dato un tag
            prefix_embeddings = self.tag_prompts[tag](bsz=batch_size)
            
            # prefix_soft_prompt + input_text
            encoder_input_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)
            
            # Update della maschera di attenzione
            prefix_attention_mask = torch.ones(batch_size, prefix_embeddings.shape[1], device=input_ids.device)
            updated_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            
            # Forward pass encoder
            encoder_output = self.encoder(
                attention_mask=updated_attention_mask,
                inputs_embeds=encoder_input_embeddings
            )
            
            # Layer di classificazione
            last_hidden_states = encoder_output.last_hidden_state
            
            # Layer di classificazione solo sulla parte del testo
            text_hidden_states = last_hidden_states[:, prefix_embeddings.shape[1]:]
            
            # Dropout e il layer lineare
            tag_scores = self.classifier(self.dropout(text_hidden_states))
            
            # Salva gli score nella matrice finale
            scores_matrix[:, :, i] = tag_scores.squeeze(-1)
            
        # SoftMax presente in nn.CrossEntropyLoss per il calcolo del loss
        # final_probabilities = torch.softmax(scores_matrix, dim=-1)
        
        return scores_matrix """
    
""" class NERPrefixTuningModelv2(nn.Module):
    def __init__(self, model_name: str, ner_tags: List[str], prefix_length: int = 10, mid_dim: int = 512):
        super().__init__()
        

        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        encoder_config = self.encoder.config
        self.ner_tags = ner_tags
        self.num_tags = len(ner_tags)
        

        self.prefix_module = NERSoftModule(
            encoder_config=encoder_config,
            prefix_length=prefix_length,
            mid_dim=mid_dim
        )
            

        self.classifier = nn.Linear(encoder_config.hidden_size, self.num_tags)
        self.dropout = nn.Dropout(encoder_config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        prefix_embeddings = self.prefix_module(batch_size)

        input_embeddings = self.encoder.embeddings(input_ids)

        encoder_input_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)

        current_batch_size = input_ids.shape[0]
        prefix_attention_mask = torch.ones(current_batch_size, self.prefix_module.preseqlen, device=input_ids.device)
        updated_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        encoder_output = self.encoder(
            attention_mask=updated_attention_mask,
            inputs_embeds=encoder_input_embeddings
        )

        last_hidden_states = encoder_output.last_hidden_state

        text_hidden_states = last_hidden_states[:, self.prefix_module.preseqlen:]

        tag_scores = self.classifier(self.dropout(text_hidden_states))
        
        return tag_scores
 """