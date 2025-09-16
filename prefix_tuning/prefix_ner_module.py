import torch
import torch.nn as nn
from transformers import PretrainedConfig

class NERPrefixModule(nn.Module):
    """
    Modulo semplificato per generare soft prompt (embedding)
    per un modello encoder (BERT..)
    """
    def __init__(self, encoder_config: PretrainedConfig, prefix_length: int, mid_dim: int):
        super().__init__()
        
        self.n_embd = encoder_config.hidden_size
        self.preseqlen = prefix_length
        self.mid_dim = mid_dim

        # Un embedding layer che rappresenta i "token virtuali" del prefix
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        
        # Un MLP (control_trans) che trasforma gli embedding del prefix
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_embd))

    def forward(self, bsz: int):
        """
        Genera il tensore di embedding del soft prompt per un dato batch.
        
        Ritorna un tensore di embedding del prompt nel formato (bsz, preseqlen, n_embd)
        """
        input_tokens = torch.arange(self.preseqlen).long().to(self.wte.weight.device)

        prompts_embedding = self.wte(input_tokens)
        prompts_embedding = self.control_trans(prompts_embedding)

        prompts_embedding = prompts_embedding.unsqueeze(0).expand(bsz, -1, -1)
        
        return prompts_embedding