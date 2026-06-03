import torch
import torch.nn as nn
from transformers import AutoModel

class NERBaselineFineTuningModel(nn.Module):
    def __init__(self, model_name: str, ner_tags: list):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        

        self.classifier = nn.Linear(self.hidden_size, len(ner_tags))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        text_hidden = out.last_hidden_state
        
        logits = self.classifier(text_hidden)
        return logits