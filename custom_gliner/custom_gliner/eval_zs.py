"""Closed-taxonomy zero-shot evaluation for soft-prompt GLiNER.

Loads a checkpoint saved by train.py and runs evaluation where every sample
sees the full label set (the protocol papers report). Two modes:

- in_domain: closed-taxonomy F1 on the JNLPBA test split.
- crossner:  closed-taxonomy F1 on each CrossNER subset (true OOD).

Both modes can be enabled at once via Cfg.mode = "both".
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Force single-GPU before any CUDA/torch init (same reason as train.py).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from gliner import GLiNER, GLiNERConfig  # noqa: E402
from gliner.data_processing.collator import SpanDataCollator  # noqa: E402
from gliner.evaluation.evaluator import BaseNEREvaluator  # noqa: E402

from soft_gliner import (  # noqa: E402
    attach_soft_prompt,
    load_trainable_state,
    make_prompt_encoder_for,
)
from utils.gliner_data_preprocessor import GlinerDataPreprocessor  # noqa: E402


# Span dataclass vs tuple compatibility (same patch as train.py).
def _gliner_get_predictions(self, ents):
    out = []
    for ent in ents:
        if hasattr(ent, "start"):
            out.append([ent.entity_type, (ent.start, ent.end)])
        else:
            out.append([ent[2], (ent[0], ent[1])])
    return out


BaseNEREvaluator.get_predictions = _gliner_get_predictions


# CrossNER multi-word labels that are stored concatenated in the CoNLL files
# (kept identical to the user's old test_ood_zs.py).
CROSSNER_LABEL_REMAP: Dict[str, str] = {
    "programlang": "programming language",
    "literarygenre": "literary genre",
    "musicalartist": "musical artist",
    "musicalinstrument": "musical instrument",
    "musicgenre": "music genre",
    "politicalparty": "political party",
    "academicjournal": "academic journal",
    "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound",
    "chemicalelement": "chemical element",
}


# ---------------------------------------------------------------------------
# Configuration — edit for your run.
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    # Match train.py architecture exactly so the saved trainable params line up.
    backbone: str = "microsoft/deberta-v3-small"
    max_width: int = 12
    max_len: int = 384
    span_mode: str = "markerV0"
    hidden_size: int = 512
    dropout: float = 0.4
    num_rnn_layers: int = 1

    # Where train.py wrote the checkpoints. Both files are evaluated by default.
    output_dir: str = "runs/soft_gliner"
    checkpoints: Tuple[str, ...] = ("best_f1.pt", "best_val_loss.pt")

    # "in_domain" | "crossner" | "both"
    mode: str = "crossner"

    # In-domain (JNLPBA test split via the user's preprocessor).
    dataset_name: str = "disi-unibo-nlp/JNLPBA"
    text_column_name: str = "tokens"
    data_dir: Optional[str] = "data"
    max_input_length: int = 512

    # CrossNER (CoNLL BIO files at <root>/<subset>/test.txt).
    crossner_root: str = "data/cross_ner"
    crossner_subsets: Tuple[str, ...] = (
        "ai", "literature", "music", "politics", "science",
    )
    crossner_remap_labels: bool = True

    # Decoding.
    threshold: float = 0.5
    batch_size: int = 12
    flat_ner: bool = True
    multi_label: bool = False

    # Mixed precision is OFF by default for eval. The model is small and the
    # closed-taxonomy pass is one-shot; running fp32 sidesteps dtype mismatches
    # at the encoder.projection layer that show up under autocast in some
    # torch/transformers/accelerate combinations. Flip to True only if you've
    # confirmed it works in your environment.
    bf16: bool = False
    fp16: bool = False


CFG = Cfg()


# ---------------------------------------------------------------------------
# Build model (mirrors train.py.build_model exactly).
# ---------------------------------------------------------------------------
def build_model() -> GLiNER:
    config = GLiNERConfig(
        model_name=CFG.backbone,
        max_width=CFG.max_width,
        max_len=CFG.max_len,
        span_mode=CFG.span_mode,
        hidden_size=CFG.hidden_size,
        dropout=CFG.dropout,
        num_rnn_layers=CFG.num_rnn_layers,
    )
    if hasattr(GLiNER, "_get_gliner_class"):
        gliner_class = GLiNER._get_gliner_class(config)
        if hasattr(gliner_class, "load_from_config"):
            return gliner_class.load_from_config(config, backbone_from_pretrained=True)
        return gliner_class(config, backbone_from_pretrained=True)
    if hasattr(GLiNER, "load_from_config"):
        return GLiNER.load_from_config(config, backbone_from_pretrained=True)
    if hasattr(GLiNER, "from_config"):
        return GLiNER.from_config(config, backbone_from_pretrained=True)
    return GLiNER(config)


def freeze_backbone(model: GLiNER) -> None:
    for p in model.model.token_rep_layer.bert_layer.parameters():
        p.requires_grad = False


def autocast_ctx():
    if CFG.fp16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if CFG.bf16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# CrossNER loader (adapted from test_ood_zs.py).
# ---------------------------------------------------------------------------
def _normalize_label(label: str) -> str:
    return label.replace("_", " ").lower()


def _bio_tags_to_spans(tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start_pos: Optional[int] = None
    entity_name: Optional[str] = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if entity_name is not None:
                spans.append((start_pos, i - 1, entity_name))
                entity_name = None
                start_pos = None
        elif tag.startswith("B-"):
            if entity_name is not None:
                spans.append((start_pos, i - 1, entity_name))
            entity_name = _normalize_label(tag[2:])
            start_pos = i
        # I- continues the current entity; nothing to do.
    if entity_name is not None:
        spans.append((start_pos, len(tokens) - 1, entity_name))
    return spans


def load_crossner_subset(root: str, subset: str, split: str = "test") -> List[dict]:
    path = os.path.join(root, subset, f"{split}.txt")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    samples: List[dict] = []
    tokens: List[str] = []
    tags: List[str] = []

    def flush():
        if tokens:
            spans = _bio_tags_to_spans(tokens, tags)
            samples.append({
                "tokenized_text": list(tokens),
                "ner": spans,
                "tags_present": sorted({s[2] for s in spans}),
            })

    for line in data.strip().split("\n"):
        if line.strip() == "":
            flush()
            tokens, tags = [], []
        else:
            parts = line.split("\t")
            tokens.append(parts[0])
            tags.append(parts[1].strip())
    flush()
    return samples


def remap_labels(samples: List[dict]) -> List[dict]:
    for s in samples:
        s["ner"] = [
            (a, b, CROSSNER_LABEL_REMAP.get(t, t)) for (a, b, t) in s["ner"]
        ]
        s["tags_present"] = sorted({x[2] for x in s["ner"]})
    return samples


def all_labels_in(samples: List[dict]) -> List[str]:
    types = set()
    for s in samples:
        for span in s["ner"]:
            types.add(span[2])
    return sorted(types)


# ---------------------------------------------------------------------------
# Closed-taxonomy eval.
# ---------------------------------------------------------------------------
def closed_taxonomy_eval(
    model: GLiNER,
    samples: List[dict],
    all_labels: List[str],
) -> Tuple[str, float]:
    """Run model on samples with the full label set forced into every prompt.

    Bypasses GLiNER's broken `model.evaluate(..., entity_types=...)` signature
    in 0.2.26 by constructing the DataLoader ourselves.
    """
    collator = SpanDataCollator(
        config=model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        return_entities=True,
        return_id_to_classes=True,
        prepare_labels=False,
    )
    types = list(all_labels)

    def collate_fn(batch):
        return collator(batch, entity_types=types)

    pred_loader = DataLoader(
        samples, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    model.eval()
    with torch.no_grad(), autocast_ctx():
        preds = model._process_batches(
            pred_loader, CFG.threshold, CFG.flat_ner, CFG.multi_label,
        )

    # Re-iterate (cheap, same shuffle=False) to gather ground-truth entities
    # in the same order as preds.
    trues: List[list] = []
    for batch in DataLoader(
        samples, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn,
    ):
        trues.extend(batch["entities"])

    out_str, f1 = BaseNEREvaluator(trues, preds).evaluate()
    return out_str, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class _Row:
    checkpoint: str
    dataset: str
    n_labels: int
    n_samples: int
    precision: float
    recall: float
    f1: float


def _parse_pr(out_str: str) -> Tuple[float, float]:
    # BaseNEREvaluator returns "P: 12.34%\tR: 56.78%\tF1: 23.45%\n"
    p, r = 0.0, 0.0
    for tok in out_str.replace("\t", " ").split():
        if tok.endswith("%") and ":" not in tok:
            continue
    for piece in out_str.split("\t"):
        piece = piece.strip()
        if piece.startswith("P:"):
            p = float(piece[2:].strip().rstrip("%")) / 100.0
        elif piece.startswith("R:"):
            r = float(piece[2:].strip().rstrip("%")) / 100.0
    return p, r


def main() -> None:
    print(f"Building GLiNER (matches train.py architecture) ...")
    # If a local backbone directory exists under the repo `data/` folder,
    # prefer that to avoid any HuggingFace downloads in offline environments.
    local_backbone = os.path.abspath(os.path.join(HERE, "..", CFG.data_dir, "microsoft--deberta-v3-base"))
    if os.path.isdir(local_backbone):
        print(f"  using local backbone at: {local_backbone}")
        CFG.backbone = local_backbone
    model = build_model()
    prompt_encoder = make_prompt_encoder_for(model)
    attach_soft_prompt(model, prompt_encoder)
    freeze_backbone(model)

    # Force every module to fp32 unless the user explicitly opted into mixed
    # precision. Eliminates dtype mismatches that appear in some torch builds
    # when autocast doesn't propagate cleanly through DataParallel/_process_batches.
    if not (CFG.bf16 or CFG.fp16):
        model.float()

    # Move to GPU if available. eval_zs.py has no HF Trainer to do this for us,
    # so without this the whole eval runs on CPU.
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    device = next(model.parameters()).device
    print(f"  device: {device}    cuda_available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(device)}")

    # In-domain samples (loaded once, regardless of which checkpoints we run).
    in_domain_samples: Optional[List[dict]] = None
    in_domain_labels: Optional[List[str]] = None
    if CFG.mode in ("in_domain", "both"):
        print("\nLoading in-domain test split (JNLPBA) ...")
        pre = GlinerDataPreprocessor(
            dataset_name=CFG.dataset_name,
            val_split_ratio=0.1,
            dataset_subset=1.0,
            convert_to_spans=True,
            filter_empty_entities=True,
            text_column_name=CFG.text_column_name,
            data_dir=CFG.data_dir,
            max_input_length=CFG.max_input_length,
        )
        in_domain_samples = pre.ds_test
        in_domain_labels = list(pre.ner_tags)
        print(f"  test samples: {len(in_domain_samples)}    labels ({len(in_domain_labels)}): {in_domain_labels}")

    # CrossNER subsets.
    crossner: Dict[str, Tuple[List[dict], List[str]]] = {}
    if CFG.mode in ("crossner", "both"):
        print("\nLoading CrossNER subsets ...")
        for subset in CFG.crossner_subsets:
            samples = load_crossner_subset(CFG.crossner_root, subset, split="test")
            if CFG.crossner_remap_labels:
                samples = remap_labels(samples)
            labels = all_labels_in(samples)
            crossner[subset] = (samples, labels)
            print(f"  {subset:<11s} samples={len(samples):>5d}  labels({len(labels)}): {labels}")

    rows: List[_Row] = []

    for ckpt_name in CFG.checkpoints:
        ckpt_path = os.path.join(CFG.output_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"\n[skip] {ckpt_path} not found.")
            continue

        print(f"\n{'='*70}\n  Loading {ckpt_name}\n{'='*70}")
        info = load_trainable_state(model, ckpt_path, device=device)
        print(
            f"  copied={info['copied']}  missing={len(info['missing'])}  "
            f"unexpected={len(info['unexpected'])}"
        )
        if info["missing"]:
            print(f"  missing examples: {info['missing'][:5]}")
        if info["unexpected"]:
            print(f"  unexpected examples: {info['unexpected'][:5]}")

        if in_domain_samples is not None:
            print(f"\n  [{ckpt_name}] in-domain (JNLPBA test, closed taxonomy)")
            out_str, f1 = closed_taxonomy_eval(model, in_domain_samples, in_domain_labels)
            p, r = _parse_pr(out_str)
            print(f"    -> {out_str.strip()}")
            rows.append(_Row(ckpt_name, "JNLPBA-test", len(in_domain_labels),
                             len(in_domain_samples), p, r, f1))

        for subset, (samples, labels) in crossner.items():
            print(f"\n  [{ckpt_name}] CrossNER/{subset} (closed taxonomy, {len(labels)} labels)")
            out_str, f1 = closed_taxonomy_eval(model, samples, labels)
            p, r = _parse_pr(out_str)
            print(f"    -> {out_str.strip()}")
            rows.append(_Row(ckpt_name, f"CrossNER/{subset}", len(labels),
                             len(samples), p, r, f1))

    # Summary.
    if rows:
        print(f"\n{'='*78}")
        print("  SUMMARY (closed-taxonomy)")
        print(f"{'='*78}")
        print(f"  {'Checkpoint':<20s} {'Dataset':<22s} {'#L':>4s} {'#N':>5s} "
              f"{'P':>8s} {'R':>8s} {'F1':>8s}")
        print(f"  {'-'*72}")
        for r in rows:
            print(
                f"  {r.checkpoint:<20s} {r.dataset:<22s} {r.n_labels:>4d} {r.n_samples:>5d} "
                f"{r.precision:>8.4f} {r.recall:>8.4f} {r.f1:>8.4f}"
            )
        print(f"{'='*78}")


if __name__ == "__main__":
    main()
