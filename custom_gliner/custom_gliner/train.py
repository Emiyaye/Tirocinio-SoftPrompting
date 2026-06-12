"""Train soft-prompt GLiNER from scratch with the DeBERTa backbone frozen.

We build a fresh GLiNER (no GLiNER checkpoint), keep the pretrained DeBERTa
weights but freeze them, attach the soft-prompt encoder, and train every other
parameter with GLiNER's own Trainer / TrainingArguments / SpanDataCollator.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Force single-GPU before any CUDA/torch init: HF Trainer auto-wraps in
# DataParallel when multiple GPUs are visible, and DataParallel does not
# propagate autocast state into the replica threads, which produces dtype
# mismatches inside the prompt encoder / projection layers. Set this env var
# before importing torch. Override by setting CUDA_VISIBLE_DEVICES yourself.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

# Allow `import soft_gliner` / `import prompt_encoder` when run from project root.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # so `utils.gliner_preprocessor` is importable

from gliner import GLiNER, GLiNERConfig  # noqa: E402
from gliner.data_processing.collator import SpanDataCollator  # noqa: E402
from gliner.training import Trainer, TrainingArguments  # noqa: E402
from gliner.evaluation.evaluator import BaseNEREvaluator  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from soft_gliner import (  # noqa: E402
    attach_soft_prompt,
    make_prompt_encoder_for,
    save_trainable_state,
)
from utils.gliner_data_preprocessor import GlinerDataPreprocessor  # noqa: E402


# GLiNER 0.2.26 inconsistency: SpanDecoder returns `Span` dataclass instances,
# but BaseNEREvaluator.get_predictions indexes them as tuples (ent[0], ent[1],
# ent[2]). Monkey-patch the evaluator to handle both forms.
def _gliner_get_predictions(self, ents):
    out = []
    for ent in ents:
        if hasattr(ent, "start"):  # Span dataclass
            out.append([ent.entity_type, (ent.start, ent.end)])
        else:  # tuple-like
            out.append([ent[2], (ent[0], ent[1])])
    return out


BaseNEREvaluator.get_predictions = _gliner_get_predictions


# ---------------------------------------------------------------------------
# Configuration — edit these for your run.
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    # Backbone / GLiNER architecture
    backbone: str = "microsoft/deberta-v3-base"
    max_width: int = 12
    max_len: int = 384
    span_mode: str = "markerV0"
    hidden_size: int = 512
    dropout: float = 0.4
    num_rnn_layers: int = 1

    # Data
    dataset_name: str = "milistu/Pile-NER-type-conll"
    data_dir: str | None = "data"             # set to a local snapshot dir if offline
    text_column_name: str = "words"
    val_split_ratio: float = 0.1
    dataset_subset: float = 1.0
    max_input_length: int = 384             # filter long sequences

    # Training (GLiNER finetune defaults)
    output_dir: str = "runs/soft_gliner"
    num_train_epochs: float = 50.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4             # for prompt encoder + secondary heads
    others_lr: float | None = None          # if set, splits encoder vs others lr
    others_weight_decay: float = 0.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    loss_reduction: str = "sum"
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    seed: int = 42

    # Evaluation (per-sample-prompt F1, GLiNER's default protocol — fast, used
    # for monitoring + checkpoint selection. Closed-taxonomy F1 is in eval_zs.py.)
    eval_threshold: float = 0.5
    eval_batch_size_for_f1: int = 12
    eval_subset: int | None = None  # if set, evaluate on this many val samples each epoch

    # Early stopping. Patience resets if EITHER val_loss or val_f1 improves.
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0

    # Optional partial unfreezing of the backbone at a specific epoch.
    # `unfreeze_backbone_at_epoch` is 1-indexed: e.g. 6 means "before the 6th epoch".
    # Set to None to keep the backbone frozen for the whole run.
    unfreeze_backbone_at_epoch: int | None = None
    unfreeze_last_n_layers: int = 3
    unfreeze_backbone_lr: float = 1e-5
    unfreeze_backbone_weight_decay: float = 0.0
    # Mixed precision is off by default. The model is small, and enabling fp16/bf16
    # with multi-GPU DataParallel produces dtype mismatches in the prompt encoder.
    # Flip these on if you've confirmed a single-GPU setup.
    bf16: bool = False
    fp16: bool = True


CFG = Cfg()


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model() -> GLiNER:
    # Prefer a local snapshot under the project's `data/` folder when available.
    model_name_ref = CFG.backbone
    if CFG.data_dir:
        project_root = os.path.dirname(HERE)
        candidate = os.path.join(project_root, CFG.data_dir, CFG.backbone.replace("/", "--"))
        if os.path.isdir(candidate):
            model_name_ref = candidate

    config = GLiNERConfig(
        model_name=model_name_ref,
        max_width=CFG.max_width,
        max_len=CFG.max_len,
        span_mode=CFG.span_mode,
        hidden_size=CFG.hidden_size,
        dropout=CFG.dropout,
        num_rnn_layers=CFG.num_rnn_layers,
    )
    # Fresh GLiNER: random GLiNER-specific weights, DeBERTa backbone loaded from HF.
    # On 0.2.x the meta `GLiNER` class auto-dispatches to a specific variant
    # (UniEncoderSpanGLiNER, ...). Caveats we work around:
    #   - `GLiNER(config)` uses `self.__class__ = type(new_instance)`, which fails on
    #     recent torch with "object layout differs".
    #   - `GLiNER.from_config(config_obj)` has a bug in 0.2.26: it only assigns its
    #     internal `config_` when given a path/dict, so passing a GLiNERConfig
    #     instance hits UnboundLocalError.
    # So we dispatch manually whenever possible.
    if hasattr(GLiNER, "_get_gliner_class"):
        gliner_class = GLiNER._get_gliner_class(config)
        if hasattr(gliner_class, "load_from_config"):
            return gliner_class.load_from_config(config, backbone_from_pretrained=True)
        return gliner_class(config, backbone_from_pretrained=True)
    if hasattr(GLiNER, "load_from_config"):
        return GLiNER.load_from_config(config, backbone_from_pretrained=True)
    if hasattr(GLiNER, "from_config"):
        return GLiNER.from_config(config, backbone_from_pretrained=True)
    # Very old API: bare constructor loads pretrained backbone automatically.
    return GLiNER(config)


def freeze_backbone(model: GLiNER) -> None:
    """Freeze the DeBERTa encoder (token_rep_layer.bert_layer); leave projection,
    span_rep_layer, prompt_rep_layer, RNN, and prompt_encoder trainable.
    """
    for p in model.model.token_rep_layer.bert_layer.parameters():
        p.requires_grad = False


def _find_encoder_layers(bert_model) -> "torch.nn.ModuleList":
    """Locate the ModuleList of transformer layers in a HF encoder.

    Works for BERT, DeBERTa-v2/v3, RoBERTa, ELECTRA, etc. (all expose
    `model.encoder.layer`). Falls back to scanning attributes if needed.
    """
    enc = getattr(bert_model, "encoder", None)
    if enc is not None and isinstance(getattr(enc, "layer", None), torch.nn.ModuleList):
        return enc.layer
    # Last resort: walk children for a ModuleList of repeated submodules.
    for _, child in bert_model.named_modules():
        if isinstance(child, torch.nn.ModuleList) and len(child) > 0:
            return child
    raise RuntimeError("Could not locate encoder.layer ModuleList on the backbone.")


class PartialBackboneUnfreezeCallback(TrainerCallback):
    """At the start of `unfreeze_at_epoch` (1-indexed), unfreeze the last
    `num_last_layers` transformer layers of the backbone, register their
    parameters as a new optimizer param group with `lr` (and matching weight
    decay), and extend the LR scheduler so it can step the new group.

    Fires at most once. Reads `optimizer` and `lr_scheduler` from the kwargs
    HF Trainer passes to `on_epoch_begin`.
    """

    def __init__(
        self,
        model,
        unfreeze_at_epoch: int,
        num_last_layers: int,
        lr: float,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.unfreeze_at_epoch = int(unfreeze_at_epoch)
        self.num_last_layers = int(num_last_layers)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.fired = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.fired:
            return control
        # state.epoch in on_epoch_begin reflects how many epochs have completed
        # (0 at the start of the very first epoch). To trigger "before the K-th
        # epoch (1-indexed)", we wait until int(state.epoch) >= K - 1.
        if int(state.epoch) < self.unfreeze_at_epoch - 1:
            return control

        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        if optimizer is None:
            print(f"  [unfreeze] no optimizer in callback kwargs; will retry next epoch")
            return control

        bert_model = self.model.model.token_rep_layer.bert_layer.model
        layer_list = _find_encoder_layers(bert_model)
        n_total = len(layer_list)
        n_unfreeze = min(self.num_last_layers, n_total)

        new_params = []
        for layer in layer_list[-n_unfreeze:]:
            for p in layer.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)

        if not new_params:
            print(f"  [unfreeze] epoch={state.epoch:.2f}: nothing to unfreeze (already trainable?)")
            self.fired = True
            return control

        optimizer.add_param_group({
            "params": new_params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        })

        # Keep the scheduler in sync so its step() doesn't index out of bounds.
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs.append(self.lr)
            if hasattr(lr_scheduler, "lr_lambdas") and lr_scheduler.lr_lambdas:
                # Reuse the same warmup/decay schedule for the new group.
                lr_scheduler.lr_lambdas.append(lr_scheduler.lr_lambdas[0])

        n_params = sum(p.numel() for p in new_params)
        print(
            f"\n  [unfreeze] epoch={state.epoch:.2f}: unfroze last {n_unfreeze}/{n_total} "
            f"backbone layers ({n_params:,} params) lr={self.lr} wd={self.weight_decay}"
        )
        self.fired = True
        return control


class BestCkptEarlyStopCallback(TrainerCallback):
    """Per-epoch: compute val F1, read val loss, save best-of-each, early-stop.

    - val F1 is GLiNER's per-sample-prompt protocol (model.evaluate).
    - val loss is read from the latest entry in state.log_history (HF logs it
      under the key "eval_loss" when eval_strategy="epoch").
    - On improvement (loss down OR f1 up, beyond min_delta) → save trainable
      state to {output_dir}/best_val_loss.pt or best_f1.pt and reset patience.
    - If neither metric improves for `patience` consecutive epochs → stop.
    """

    def __init__(
        self,
        model,
        eval_data,
        output_dir: str,
        batch_size: int,
        threshold: float,
        patience: int,
        min_delta: float = 0.0,
    ):
        self.model = model
        self.eval_data = eval_data
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.threshold = threshold
        self.patience = patience
        self.min_delta = min_delta

        self.best_val_loss = float("inf")
        self.best_val_f1 = float("-inf")
        self.epochs_without_improvement = 0

        os.makedirs(output_dir, exist_ok=True)

    def _autocast_ctx(self, args):
        from contextlib import nullcontext
        if getattr(args, "fp16", False):
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if getattr(args, "bf16", False):
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _latest_eval_loss(state):
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                return float(entry["eval_loss"])
        return None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Fires AFTER HF's _maybe_log_save_evaluate has populated state.log_history
        # with eval_loss for the current epoch. With eval_strategy="epoch" this is
        # once per epoch, right after HF prints the "Epoch | Training | Val" row.
        # (If we used on_epoch_end, val_loss in log_history would still be from
        #  the previous epoch and the printout would look one-epoch-shifted.)
        was_training = self.model.training

        with self._autocast_ctx(args):
            out_str, val_f1 = self.model.evaluate(
                self.eval_data, batch_size=self.batch_size, threshold=self.threshold
            )

        # Prefer the metrics dict HF passes in (freshest), then fall back to
        # log_history scanning.
        val_loss = None
        if metrics is not None:
            val_loss = metrics.get("eval_loss")
        if val_loss is None:
            val_loss = self._latest_eval_loss(state)

        improved_loss = val_loss is not None and val_loss < self.best_val_loss - self.min_delta
        improved_f1 = val_f1 > self.best_val_f1 + self.min_delta

        saved = []
        if improved_loss:
            self.best_val_loss = val_loss
            save_trainable_state(self.model, os.path.join(self.output_dir, "best_val_loss.pt"))
            saved.append("best_val_loss.pt")
        if improved_f1:
            self.best_val_f1 = val_f1
            save_trainable_state(self.model, os.path.join(self.output_dir, "best_f1.pt"))
            saved.append("best_f1.pt")

        if improved_loss or improved_f1:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        loss_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(
            f"\n[epoch {state.epoch:.2f}] val_loss={loss_str}  "
            f"val_f1={val_f1:.4f}  patience={self.epochs_without_improvement}/{self.patience}  "
            f"saved={saved or '-'}  ({out_str.strip()})"
        )

        if self.epochs_without_improvement >= self.patience:
            print(f"  early stopping: no improvement in {self.patience} epoch(s).")
            control.should_training_stop = True

        if was_training:
            self.model.train()
        return control


def report_params(model: GLiNER) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params:     {total:,}")
    print(f"  trainable params: {trainable:,}  ({100 * trainable / total:.2f}%)")
    by_module = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = ".".join(n.split(".")[:3])
        by_module[head] = by_module.get(head, 0) + p.numel()
    print("  trainable param breakdown (top-3 components):")
    for k, v in sorted(by_module.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<50s} {v:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(CFG.seed)

    print("Loading dataset ...")
    pre = GlinerDataPreprocessor(
        dataset_name=CFG.dataset_name,
        val_split_ratio=CFG.val_split_ratio,
        dataset_subset=CFG.dataset_subset,
        convert_to_spans=True,
        filter_empty_entities=True,
        text_column_name=CFG.text_column_name,
        data_dir=CFG.data_dir,
        max_input_length=CFG.max_input_length,
    )
    train_ds = pre.ds_train
    eval_ds = pre.ds_val
    print(f"  train: {len(train_ds)}    val: {len(eval_ds)}    types: {len(pre.ner_tags)}")

    print("\nBuilding GLiNER from scratch (DeBERTa backbone pretrained, GLiNER heads random) ...")
    model = build_model()

    print("Attaching soft prompt encoder ...")
    prompt_encoder = make_prompt_encoder_for(model)
    attach_soft_prompt(model, prompt_encoder)

    print("Freezing DeBERTa backbone ...")
    freeze_backbone(model)
    report_params(model)

    data_collator = SpanDataCollator(
        config=model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        others_lr=CFG.others_lr,
        others_weight_decay=CFG.others_weight_decay,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler_type,
        focal_loss_alpha=CFG.focal_loss_alpha,
        focal_loss_gamma=CFG.focal_loss_gamma,
        loss_reduction=CFG.loss_reduction,
        logging_steps=CFG.logging_steps,
        evaluation_strategy=CFG.eval_strategy,
        save_strategy="no",  # We save best ckpts ourselves in BestCkptEarlyStopCallback.
        seed=CFG.seed,
        bf16=CFG.bf16,
        fp16=CFG.fp16,
        remove_unused_columns=False,        # collator consumes raw 'tokenized_text' / 'ner'
        dataloader_num_workers=0,
        report_to=[],
    )

    per_epoch_eval = (
        eval_ds if CFG.eval_subset is None else eval_ds[: CFG.eval_subset]
    )
    callbacks = [
        BestCkptEarlyStopCallback(
            model=model,
            eval_data=per_epoch_eval,
            output_dir=CFG.output_dir,
            batch_size=CFG.eval_batch_size_for_f1,
            threshold=CFG.eval_threshold,
            patience=CFG.early_stopping_patience,
            min_delta=CFG.early_stopping_min_delta,
        )
    ]

    if CFG.unfreeze_backbone_at_epoch is not None:
        callbacks.append(
            PartialBackboneUnfreezeCallback(
                model=model,
                unfreeze_at_epoch=CFG.unfreeze_backbone_at_epoch,
                num_last_layers=CFG.unfreeze_last_n_layers,
                lr=CFG.unfreeze_backbone_lr,
                weight_decay=CFG.unfreeze_backbone_weight_decay,
            )
        )
        print(
            f"  partial-unfreeze callback armed: "
            f"unfreeze last {CFG.unfreeze_last_n_layers} backbone layers at epoch "
            f"{CFG.unfreeze_backbone_at_epoch} (lr={CFG.unfreeze_backbone_lr})"
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"\nStarting training for up to {CFG.num_train_epochs} epochs "
          f"(early stop patience={CFG.early_stopping_patience}) ...")
    trainer.train()
    print(
        f"\nTraining done. Best val_loss={callbacks[0].best_val_loss:.4f}, "
        f"best val_f1={callbacks[0].best_val_f1:.4f}.\n"
        f"Checkpoints saved to {CFG.output_dir}/best_val_loss.pt and {CFG.output_dir}/best_f1.pt.\n"
        f"Run eval_zs.py for closed-taxonomy F1 (in-domain test or CrossNER OOD)."
    )


if __name__ == "__main__":
    main()
