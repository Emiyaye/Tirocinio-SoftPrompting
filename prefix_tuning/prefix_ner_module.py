import torch
import torch.nn as nn
from transformers import PretrainedConfig

class NERPrefixModule(nn.Module):
    """
    Modulo per generare Key e Value soft prompt per ogni layer 
    per un modello encoder (BERT..)
    """
    def __init__(self, encoder_config: PretrainedConfig, prefix_length: int, mid_dim: int):
        super().__init__()
        
        self.n_embd = encoder_config.hidden_size
        self.preseqlen = prefix_length
        self.mid_dim = mid_dim

        self.num_layers = encoder_config.num_hidden_layers
        self.num_heads = encoder_config.num_attention_heads
        self.head_dim = self.n_embd // self.num_heads 

        # Un embedding layer che rappresenta i "token virtuali" del prefix
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)

        # Un MLP (control_trans)
        # Output : preseqlen * (num_layers * 2 * n_embd)
        target_dim = self.num_layers * 2 * self.n_embd
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            #nn.GELU(), 
            nn.Linear(self.mid_dim, target_dim),
            #nn.LayerNorm(target_dim)
            )
            

    def forward(self, bsz: int):
        """
        Genera i tensori Key e Value per past_key_values.
        
        Ritorna una tupla di tensori Key Value nel formato 
        ((2, bsz, n_head, preseqlen, n_embd) x num_layers)
        """
        input_tokens = torch.arange(self.preseqlen).long().to(self.wte.weight.device)

        prompts_embedding = self.wte(input_tokens) # (preseqlen, n_embd)
        prompts_kv = self.control_trans(prompts_embedding) # (preseqlen, num_layers * 2 * n_embd)

        # Shape: (preseqlen, num_layers, 2, num_heads, head_dim)
        prompts_kv = prompts_kv.view(
            self.preseqlen, 
            self.num_layers, 
            2, 
            self.num_heads, 
            self.head_dim
        )


        # Target: (num_layers, 2, bsz, num_heads, preseqlen, head_dim)
        prompts_kv = prompts_kv.permute(1, 2, 0, 3, 4) # (num_layers, 2, preseqlen, num_heads, head_dim)
        prompts_kv = prompts_kv.unsqueeze(2).expand(-1, -1, bsz, -1, -1, -1)
        
        # past_key_values = tupla di tuple, dove ogni elemento è (key, value)
        # key.shape = (bsz, num_heads, preseqlen, head_dim)
        # value.shape = (bsz, num_heads, preseqlen, head_dim)
        past_key_values = []
        for layer in prompts_kv:
            # layer.shape = (2, bsz, num_heads, preseqlen, head_dim)
            # Trasponiamo in (bsz, num_heads, preseqlen, head_dim)
            key, value = layer[0].permute(0, 2, 1, 3), layer[1].permute(0, 2, 1, 3) 
            past_key_values.append((key, value))
            
        return tuple(past_key_values)