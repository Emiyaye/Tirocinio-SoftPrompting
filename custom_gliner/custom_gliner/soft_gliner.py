"""Attach a soft prompt encoder to a pretrained GLiNER model.

The prompt encoder perturbs the *input embeddings* of entity-name subwords before
they reach the transformer, leaving every other token (and GLiNER's tokenization,
collation, and downstream architecture) untouched.
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gliner.modeling.utils import extract_prompt_features_and_word_embeddings

from prompt_encoder import _PromptEncoderBlock


class PromptEncoder(nn.Module):
    """Single-block self-conditioned prompt encoder.

    Output replaces (residually) the entity-name subword input embeddings.
    """

    def __init__(self, hidden_size: int, attention_heads: int, dropout: float = 0.1):
        super().__init__()
        self.block = _PromptEncoderBlock(
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, kpm: Optional[torch.Tensor]) -> torch.Tensor:
        # Self-conditioned: kv = x.
        return self.block(x, kv=x, kv_key_padding=kpm)


def _find_entity_spans(
    input_ids: torch.Tensor, ent_id: int, sep_id: int
) -> List[List[Tuple[int, int]]]:
    """For each row, return per-entity (start, end) name-subword spans.

    Span boundaries: between consecutive [ENT] tokens, and from the last [ENT]
    up to the first [SEP] following it.
    """
    B, L = input_ids.shape
    out: List[List[Tuple[int, int]]] = []
    for b in range(B):
        ids = input_ids[b]
        ent_pos = (ids == ent_id).nonzero(as_tuple=True)[0].tolist()
        if not ent_pos:
            out.append([])
            continue
        sep_after = (ids == sep_id).nonzero(as_tuple=True)[0]
        sep_after = sep_after[sep_after > ent_pos[-1]]
        end_of_prompt = sep_after[0].item() if len(sep_after) > 0 else L

        spans: List[Tuple[int, int]] = []
        for i, p in enumerate(ent_pos):
            start = p + 1
            end = ent_pos[i + 1] if i + 1 < len(ent_pos) else end_of_prompt
            if end > start:
                spans.append((start, end))
        out.append(spans)
    return out


def _pack(
    embeds: torch.Tensor, spans_per_row: List[List[Tuple[int, int]]]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[Tuple[int, int, int]]]:
    """Pack per-entity name-subword embeddings into (E, K_max, D) with key-padding mask.

    Returns (packed, kpm, index) where index[i] = (batch_row, start, length).
    kpm follows nn.MultiheadAttention convention: True at PAD positions.
    """
    D = embeds.size(-1)
    flat: List[Tuple[int, int, int]] = []
    for b, spans in enumerate(spans_per_row):
        for s, e in spans:
            flat.append((b, s, e - s))
    if not flat:
        return None, None, flat
    K_max = max(L for _, _, L in flat)
    E = len(flat)
    packed = torch.zeros(E, K_max, D, dtype=embeds.dtype, device=embeds.device)
    kpm = torch.ones(E, K_max, dtype=torch.bool, device=embeds.device)
    for i, (b, s, L) in enumerate(flat):
        packed[i, :L] = embeds[b, s : s + L]
        kpm[i, :L] = False
    return packed, kpm, flat


def _scatter_back(
    embeds: torch.Tensor,
    perturbed: torch.Tensor,
    flat: List[Tuple[int, int, int]],
) -> torch.Tensor:
    out = embeds.clone()
    for i, (b, s, L) in enumerate(flat):
        out[b, s : s + L] = perturbed[i, :L]
    return out


def _zero_padded_prompts_hook(module, inputs, output):
    """Re-zero prompt embeddings at padded class slots.

    `extract_prompt_features` zero-pads `prompts_embedding` for class slots beyond
    the sample's actual count. `prompt_rep_layer` (Linear with bias) then turns
    those zeros into non-zero vectors, which produces real scores at padded class
    positions and makes the decoder index into out-of-range `id_to_class` keys.

    The hook detects padded rows by exact-zero input and zeroes their projected
    output, so the einsum at those slots is guaranteed to be zero.
    """
    pad_mask = (inputs[0].abs().sum(dim=-1) == 0)  # (B, C)
    if pad_mask.any():
        output = output.clone()
        output[pad_mask] = 0
    return output


def attach_soft_prompt(gliner_model: Any, prompt_encoder: PromptEncoder) -> None:
    """Monkey-patch ``get_representations`` on the inner model so input embeddings
    of entity-name subwords are perturbed by ``prompt_encoder`` before encoding.
    Also installs a forward hook on ``prompt_rep_layer`` that keeps padded class
    slots at zero, so the decoder never sees fake scores beyond the sample's
    actual entity-type count.
    """
    inner = gliner_model.model
    tokenizer = gliner_model.data_processor.transformer_tokenizer
    sep_id = tokenizer.convert_tokens_to_ids(gliner_model.config.sep_token)

    inner.add_module("prompt_encoder", prompt_encoder)
    if hasattr(inner, "prompt_rep_layer"):
        inner.prompt_rep_layer.register_forward_hook(_zero_padded_prompts_hook)

    def soft_get_representations(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        words_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        embed_layer = self.token_rep_layer.bert_layer.model.get_input_embeddings()
        embeds = embed_layer(input_ids)

        spans = _find_entity_spans(input_ids, self.config.class_token_index, sep_id)
        packed, kpm, flat = _pack(embeds, spans)
        if packed is not None:
            # Run the prompt encoder in its own parameter dtype, then cast back.
            # Guards against autocast/DataParallel mismatches where embeds may be
            # in a different precision than the prompt encoder's weights.
            pe_dtype = next(self.prompt_encoder.parameters()).dtype
            perturbed = self.prompt_encoder(packed.to(pe_dtype), kpm)
            embeds = _scatter_back(embeds, perturbed.to(embeds.dtype), flat)

        encoder_kwargs = dict(kwargs)
        encoder_kwargs["inputs_embeds"] = embeds
        token_embeds = self.token_rep_layer(None, attention_mask, **encoder_kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            extract_prompt_features_and_word_embeddings(
                self.config.class_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                text_lengths,
                words_mask,
                self.config.embed_ent_token,
            )
        )
        if hasattr(self, "rnn"):
            words_embedding = self.rnn(words_embedding, mask)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    inner.get_representations = types.MethodType(soft_get_representations, inner)


def save_trainable_state(model: Any, path: str) -> None:
    """Save only parameters with requires_grad=True (skips frozen DeBERTa)."""
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path)


def load_trainable_state(model: Any, path: str, device: Optional[Any] = None) -> Dict[str, int]:
    """Copy parameters from a checkpoint into matching named parameters of model.

    Mirrors the pattern from the user's old test_ood_zs.py. Returns a small dict
    summarizing what was copied vs missing vs unexpected for sanity logging.
    """
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    name_to_param = dict(model.named_parameters())

    copied = 0
    missing = []
    for n, p in name_to_param.items():
        if n in state:
            p.data.copy_(state[n].to(p.device, dtype=p.dtype))
            copied += 1
        elif p.requires_grad:
            missing.append(n)

    unexpected = [k for k in state if k not in name_to_param]
    return {"copied": copied, "missing": missing, "unexpected": unexpected}


def make_prompt_encoder_for(gliner_model: Any) -> PromptEncoder:
    """Build a PromptEncoder with dims matching the underlying BERT input space."""
    bert_cfg = gliner_model.model.token_rep_layer.bert_layer.model.config
    hidden = bert_cfg.hidden_size
    heads = getattr(bert_cfg, "num_attention_heads", 8)
    if hidden % heads != 0:
        for h in (8, 4, 2, 1):
            if hidden % h == 0:
                heads = h
                break
    return PromptEncoder(hidden_size=hidden, attention_heads=heads)
