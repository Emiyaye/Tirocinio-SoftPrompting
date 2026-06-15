"""Explainability: decode the soft-prompt-perturbed entity tags back into words.

WHAT THIS DOES (the short version)
----------------------------------
In this custom GLiNER, the soft prompt encoder does *not* add tokens to the
prompt. Instead it **perturbs the input embeddings of the entity-tag subwords**
(the word-pieces that come right after each [ENT] marker) *before* they enter the
transformer. See `soft_gliner.soft_get_representations`.

Because those perturbed vectors live in the SAME space as the model's vocabulary
embedding matrix, we can ask a very simple, very visual question:

    After the soft prompt perturbs a tag like "location", which real vocabulary
    words is its embedding now closest to (by cosine similarity)?

If "location" drifts toward words like "city", "place", "region", that is
human-readable evidence the soft prompt is moving the tag toward a meaningful
part of the embedding space.

HOW (the pipeline, all in `main`)
---------------------------------
  1. Rebuild the exact same GLiNER + soft prompt as training, load the checkpoint.
  2. Take ONE example sentence + a few tags and let GLiNER build its input_ids
     (the "[ENT] tag1 [ENT] tag2 ... [SEP] sentence" format) via its own collator.
  3. Look up the original input embeddings of the tag subwords.
  4. Run the prompt encoder on them -> perturbed embeddings.
  5. For each perturbed tag, cosine-compare it against the whole vocabulary and
     print the closest words.

Run it:
    python explain_tag_cosine.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import string

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make `soft_gliner` / `prompt_encoder` importable when run from this folder.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from gliner import GLiNER, GLiNERConfig  # noqa: E402
from gliner.data_processing.collator import SpanDataCollator  # noqa: E402

# We reuse the *exact* helpers training uses, so the perturbation here is
# identical to the one inside the model. No re-implementation, no drift.
from soft_gliner import (  # noqa: E402
    attach_soft_prompt,
    make_prompt_encoder_for,
    load_trainable_state,
    _find_entity_spans,   # find the (start, end) subword span of each tag
    _pack,                # pack tag spans into a (n_tags, K, D) tensor
    _scatter_back,        # write perturbed vectors back into the embedding tensor
)


# ---------------------------------------------------------------------------
# Settings — must match the architecture the checkpoint was trained with
# (train.py used microsoft/deberta-v3-base, hidden_size=512).
# ---------------------------------------------------------------------------
BACKBONE = "microsoft/deberta-v3-small"
HIDDEN_SIZE = 512
MAX_WIDTH = 12
MAX_LEN = 384
SPAN_MODE = "markerV0"
DROPOUT = 0.4
NUM_RNN_LAYERS = 1

# Checkpoint to explain. best_f1.pt is the one selected by validation F1.
CHECKPOINT = os.path.join(HERE, "soft_gliner_1", "best_f1.pt")

# How many nearest vocabulary words to print per tag.
TOP_K = 10

# ---------------------------------------------------------------------------
# CrossNER split definitions
# Each entry: (domain_name, list_of_entity_tags, representative_sentence)
# The sentence is intentionally generic but domain-flavoured so the tokenizer
# produces a realistic context around the entity spans.
# ---------------------------------------------------------------------------
CROSSNER_SPLITS = [
    (
        "ai",
        [
            "algorithm", "conference", "else", "field", "metrics",
            "organisation", "person", "product", "programlang",
            "researcher", "task", "university",
        ],
        (
            "The researcher presented a novel neural network algorithm at the "
            "NeurIPS conference, achieving state-of-the-art metrics on the "
            "image classification task."
        ),
    ),
    (
        "literature",
        [
            "award", "book", "country", "else", "event", "literarygenre",
            "magazine", "misc", "organisation", "person", "poem", "writer",
        ],
        (
            "The writer received the Booker Award for her novel set in France, "
            "a work blending the literary genre of magical realism with poetry."
        ),
    ),
    (
        "music",
        [
            "album", "award", "band", "country", "else", "event", "misc",
            "musicalartist", "musicalinstrument", "musicgenre",
            "organisation", "person", "song",
        ],
        (
            "The band released their debut album and won a Grammy Award at the "
            "festival, blending jazz and rock genres with saxophone solos."
        ),
    ),
    (
        "politics",
        [
            "country", "election", "else", "event", "misc", "organisation",
            "person", "politician", "politicalparty",
        ],
        (
            "The politician won the general election representing the Labour "
            "Party and signed a treaty between France and Germany."
        ),
    ),
    (
        "science",
        [
            "academicjournal", "award", "chemicalelement", "chemicalcompound",
            "country", "discipline", "else", "enzyme", "event", "misc",
            "organisation", "person", "protein", "scientist",
            "theory", "university",
        ],
        (
            "The scientist published findings on the protein structure of an "
            "enzyme in the journal Nature, advancing the theory of molecular "
            "biology at MIT."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers (identical to explain_tag_cosine.py)
# ---------------------------------------------------------------------------

def _freeze_backbone(model) -> None:
    for p in model.model.token_rep_layer.bert_layer.parameters():
        p.requires_grad = False


def build_model() -> GLiNER:
    """Rebuild GLiNER exactly like train.py / eval_zs.py do."""
    # Prefer a local backbone snapshot under ../data if present (offline-friendly).
    model_name = BACKBONE
    local = os.path.join(os.path.dirname(HERE), "data", BACKBONE.replace("/", "--"))
    if os.path.isdir(local):
        model_name = local

    config = GLiNERConfig(
        model_name=model_name,
        max_width=MAX_WIDTH,
        max_len=MAX_LEN,
        span_mode=SPAN_MODE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        num_rnn_layers=NUM_RNN_LAYERS,
    )
   # Same defensive dispatch as train.py (handles GLiNER 0.2.x quirks).
    if hasattr(GLiNER, "_get_gliner_class"):
        cls = GLiNER._get_gliner_class(config)
        if hasattr(cls, "load_from_config"):
            return cls.load_from_config(config, backbone_from_pretrained=True)
        return cls(config, backbone_from_pretrained=True)
    if hasattr(GLiNER, "load_from_config"):
        return GLiNER.load_from_config(config, backbone_from_pretrained=True)
    return GLiNER(config)


def build_valid_vocab(tokenizer, vocab_size, device):
    """Keep only whole-word, non-special, non-punctuation tokens."""
    punct_set = set(string.punctuation)
    valid_indices, valid_tokens = [], []

    for idx in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token in [
            tokenizer.cls_token, tokenizer.sep_token,
            tokenizer.mask_token, tokenizer.pad_token, tokenizer.unk_token,
        ]:
            continue
        if token.startswith("[unused") or token.startswith("<") or token.endswith(">"):
            continue
        if not token.startswith("▁"):
            continue
        bare = token[1:]
        if not bare:
            continue
        if len(bare) == 1 and bare in punct_set:
            continue
        if all(c in punct_set for c in bare):
            continue
        valid_indices.append(idx)
        valid_tokens.append(bare)

    valid_indices = torch.tensor(valid_indices, device=device)
    return valid_indices, valid_tokens


def nearest_words(vec, norm_vocab, valid_indices, valid_tokens, top_k, exclude_ids=()):
    """Return the `top_k` *filtered* vocabulary tokens most cosine-similar to `vec`.
 
    `vec`           : (D,) a single embedding (e.g. a perturbed tag subword).
    `norm_vocab`    : (V, D) the full vocabulary embedding matrix, L2-normalized.
    `valid_indices` : (V_filtered,) ids of tokens kept after filtering.
    `valid_tokens`  : list[str], display strings aligned with `valid_indices`.
    """
    v = F.normalize(vec, dim=-1)                       # (D,)
    sims = norm_vocab[valid_indices] @ v               # (V_filtered,)
    for i in exclude_ids:
        match = (valid_indices == i).nonzero(as_tuple=True)[0]
        if len(match) > 0:
            sims[match[0]] = -1.0
    values, local_idx = sims.topk(top_k)
    tokens = [valid_tokens[i] for i in local_idx.tolist()]
    return list(zip(tokens, values.tolist()))


def fmt(neighbours):
    return " | ".join(f"{w}({s:.2f})" for w, s in neighbours)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- 1. Build model + soft prompt, load checkpoint ---------------------
    print("Building GLiNER + soft prompt ...")
    model = build_model()
    prompt_encoder = make_prompt_encoder_for(model)
    attach_soft_prompt(model, prompt_encoder)
    _freeze_backbone(model)
    model.float().to(device).eval()

    print(f"Loading checkpoint: {CHECKPOINT}")
    info = load_trainable_state(model, CHECKPOINT, device=device)
    print(f"  copied={info['copied']}  missing={len(info['missing'])}  "
          f"unexpected={len(info['unexpected'])}\n")

    tokenizer        = model.data_processor.transformer_tokenizer
    class_token_index = model.config.class_token_index
    sep_id           = tokenizer.convert_tokens_to_ids(model.config.sep_token)

    # The vocabulary embedding matrix (V, D): every row is a real word's vector.
    embed_layer = model.model.token_rep_layer.bert_layer.model.get_input_embeddings()
    vocab_weight = embed_layer.weight.detach().to(device)        # (V, D)
    norm_vocab = F.normalize(vocab_weight, dim=-1)               # normalize once
    
    
    # Filter out specials / sub-word pieces / punctuation
    valid_indices, valid_tokens = build_valid_vocab(
        tokenizer, vocab_weight.size(0), device
    )

    vocab_norm_mean = vocab_weight.norm(dim=-1).mean().item()

    collator = SpanDataCollator(
        config=model.config,
        data_processor=model.data_processor,
        prepare_labels=False,
    )

    # --- 2. Iterate over CrossNER splits -----------------------------------
    for domain, tags, sentence in CROSSNER_SPLITS:
        print("\n" + "=" * 80)
        print(f"  DOMAIN : {domain.upper()}")
        print(f"  TAGS   : {tags}")
        print(f"  SENTENCE: {sentence}")
        print("=" * 80)

        tokens = sentence.split()          # simple whitespace tokenisation
        sample = {"tokenized_text": tokens, "ner": []}
        batch  = collator([sample], entity_types=tags)
        input_ids = batch["input_ids"].to(device)   # (1, L)

        with torch.no_grad():
            embeds = embed_layer(input_ids)          # (1, L, D)

            spans = _find_entity_spans(input_ids, class_token_index, sep_id)
            if not spans or len(spans[0]) == 0:
                print("  [!] No entity spans found — skipping.")
                continue

            packed, kpm, flat = _pack(embeds, spans)
            pe_dtype  = next(model.model.prompt_encoder.parameters()).dtype
            perturbed = model.model.prompt_encoder(packed.to(pe_dtype), kpm)
            perturbed_embeds = _scatter_back(
                embeds, perturbed.to(embeds.dtype), flat
            )

        # --- 3. Per-token cosine probe ------------------------------------
        print(f"\n  Reference: mean ||vocab emb|| = {vocab_norm_mean:.3f}\n")

        for i in range(input_ids.size(1)):
            token_id   = input_ids[0, i].item()
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            
            # Skip padding
            if token_id == tokenizer.pad_token_id:
                continue

            # One vector per tag = mean over its subword pieces.
            original_repr = embeds[0, i]            # before soft prompt
            perturbed_repr = perturbed_embeds[0, i]  # AFTER soft prompt

            # How far did the soft prompt move the tag? 1.0 = unchanged, ~0 = orthogonal.
            displacement = F.cosine_similarity(original_repr, perturbed_repr, dim=0).item()
            
            # filter unperturbed
            if displacement > 0.99:
                continue

            orig_norm = original_repr.norm().item()
            pert_norm = perturbed_repr.norm().item()
            delta     = perturbed_repr - original_repr
            delta_norm = delta.norm().item()

            orig_nbrs = nearest_words(
                original_repr, norm_vocab, valid_indices, valid_tokens, TOP_K
            )
            pert_nbrs = nearest_words(
                perturbed_repr, norm_vocab, valid_indices, valid_tokens, TOP_K
            )

            print(f"  Token [{i:3d}] \"{token_text}\"  "
                  f"cosine(orig,pert) = {displacement:.3f}")
            print(f"    ||orig|| = {orig_norm:.3f}   ||pert|| = {pert_norm:.3f}   "
                  f"||delta|| = {delta_norm:.3f}")
            print(f"    before : {fmt(orig_nbrs)}")
            print(f"    after  : {fmt(pert_nbrs)}")

            if delta_norm > 1e-4:
                delta_nbrs = nearest_words(
                    delta, norm_vocab, valid_indices, valid_tokens, TOP_K
                )
                print(f"    delta  : {fmt(delta_nbrs)}")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()