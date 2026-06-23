from pathlib import Path
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities

def forward_custom(model, input_ids, attention_mask,
                   prefix_override=None, output_attentions=False):
    """
    Run the model, optionally replacing the prefix with a given one.
    """
    bsz = input_ids.shape[0]
    if prefix_override is None:
        prefix = model.prefix_module(bsz=bsz)
    else:
        prefix = prefix_override.expand(bsz, -1, -1)

    inp_emb = model.encoder.get_input_embeddings()(input_ids)
    enc_in  = torch.cat([prefix, inp_emb], dim=1)

    pref_mask = torch.ones(bsz, prefix.shape[1], device=input_ids.device)
    full_mask = torch.cat([pref_mask, attention_mask], dim=1)

    out = model.encoder(inputs_embeds=enc_in,
                        attention_mask=full_mask,
                        output_attentions=output_attentions)
    text_hidden = out.last_hidden_state[:, prefix.shape[1]:]
    logits = model.classifier(text_hidden)
    return logits, out


def compute_f1(model, loader, device, id_to_tag,
               prefix_override=None, print_report=False):

    model.eval()
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn      = batch['attention_mask'].to(device)
            labels    = batch['labels'].to(device)

            logits, _ = forward_custom(model, input_ids, attn,
                                       prefix_override=prefix_override)
            preds = logits.argmax(-1)

            for i in range(labels.shape[0]):
                t_tags, p_tags = [], []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        t_tags.append(id_to_tag[labels[i, j].item()])
                        p_tags.append(id_to_tag[preds[i, j].item()])
                true_labels.append(t_tags)
                predicted_labels.append(p_tags)

    if print_report:
        print(classification_report(true_labels, predicted_labels, digits=4))

    current_f1 = f1_score(true_labels, predicted_labels)
    return current_f1

def _convert_loader_to_list(loader):
    batches = []
    for batch in loader:
        batches.append(batch)
    return batches

def _list_to_loader(batches):
    class BatchIterator:
        def __init__(self, batch_list):
            self.batches = batch_list
        def __iter__(self):
            return iter(self.batches)
    return BatchIterator(batches)


def forward_with_prefix(model, input_ids, attention_mask, prefix_override=None, output_attentions=False):
    bsz = input_ids.shape[0]
    if prefix_override is None:
        prefix = model.prefix_module(bsz=bsz)
    else:
        prefix = prefix_override.expand(bsz, -1, -1)

    input_embeddings = model.encoder.get_input_embeddings()(input_ids)
    encoder_inputs = torch.cat([prefix, input_embeddings], dim=1)

    prefix_mask = torch.ones(bsz, prefix.shape[1], device=input_ids.device)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    encoder_output = model.encoder(
        inputs_embeds=encoder_inputs,
        attention_mask=full_mask,
        output_attentions=output_attentions,
    )
    text_hidden = encoder_output.last_hidden_state[:, prefix.shape[1]:]
    logits = model.classifier(text_hidden)
    return logits, encoder_output


def collect_predictions(model, loader, device, id_to_tag, prefix_override=None, max_batches=None):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = forward_with_prefix(
                model,
                input_ids,
                attention_mask,
                prefix_override=prefix_override,
            )
            predictions = logits.argmax(dim=-1)

            for i in range(labels.shape[0]):
                gold_sentence = []
                pred_sentence = []
                for j in range(labels.shape[1]):
                    if labels[i, j].item() != -100:
                        gold_sentence.append(id_to_tag[labels[i, j].item()])
                        pred_sentence.append(id_to_tag[predictions[i, j].item()])
                true_labels.append(gold_sentence)
                predicted_labels.append(pred_sentence)

    return true_labels, predicted_labels


def per_class_report(model, loader, device, id_to_tag, prefix_override=None, max_batches=None):
    y_true, y_pred = collect_predictions(
        model,
        loader,
        device,
        id_to_tag,
        prefix_override=prefix_override,
        max_batches=max_batches,
    )
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "label": label,
                    "precision": values.get("precision", np.nan),
                    "recall": values.get("recall", np.nan),
                    "f1": values.get("f1-score", np.nan),
                    "support": values.get("support", np.nan),
                }
            )
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    display(df)
    return df, y_true, y_pred


def entity_error_analysis(y_true, y_pred):
    rows = []
    for sent_idx, (gold_tags, pred_tags) in enumerate(zip(y_true, y_pred)):
        gold_entities = set(get_entities(gold_tags))
        pred_entities = set(get_entities(pred_tags))

        for entity in gold_entities - pred_entities:
            label, start, end = entity
            rows.append(
                {
                    "sentence": sent_idx,
                    "error": "false_negative",
                    "label": label,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                }
            )

        for entity in pred_entities - gold_entities:
            label, start, end = entity
            rows.append(
                {
                    "sentence": sent_idx,
                    "error": "false_positive",
                    "label": label,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                }
            )

    errors = pd.DataFrame(rows)
    if errors.empty:
        print("Nessun errore entity-level trovato.")
        return errors

    summary = (
        errors.groupby(["error", "label", "length"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    display(summary.head(30))

    plt.figure(figsize=(10, 4))
    sns.countplot(data=errors, x="length", hue="error")
    plt.title("Errori entity-level per lunghezza dell'entita")
    plt.xlabel("Lunghezza entita in token")
    plt.ylabel("Numero errori")
    plt.tight_layout()
    plt.show()
    return errors


def trainable_parameter_report(model):
    rows = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params
        if param.requires_grad:
            trainable_params += n_params
            rows.append(
                {
                    "module": name.split(".")[0],
                    "parameter": name,
                    "n_params": n_params,
                }
            )

    df = pd.DataFrame(rows)
    by_module = df.groupby("module", as_index=False)["n_params"].sum()
    by_module["share_of_trainable"] = by_module["n_params"] / trainable_params

    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri addestrabili: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    display(by_module.sort_values("n_params", ascending=False))
    return by_module


def load_peft_results(results_dir="results/final"):
    base = Path(results_dir)
    if not (base / "testing_results.csv").exists():
        base = Path(__file__).resolve().parent / results_dir
    prefix = pd.read_csv(base / "testing_results.csv", sep=";")
    soft = pd.read_csv(base / "soft_testing_results.csv", sep=";")
    prefix["Method"] = "prefix"
    soft["Method"] = "soft_prompt"
    return pd.concat([prefix, soft], ignore_index=True)


def compare_peft_results(results_dir="results/final"):
    df = load_peft_results(results_dir)
    best = (
        df.sort_values("Micro Avg F1-Score", ascending=False)
        .groupby(["Method", "Model", "Dataset"], as_index=False)
        .first()
    )
    display(best.sort_values(["Dataset", "Model", "Method"]))

    pivot = best.pivot_table(
        index=["Model", "Dataset"],
        columns="Method",
        values="Micro Avg F1-Score",
    )
    if {"prefix", "soft_prompt"}.issubset(pivot.columns):
        pivot["prefix_minus_soft"] = pivot["prefix"] - pivot["soft_prompt"]
    display(pivot.sort_values("prefix_minus_soft", ascending=False))
    return df, best, pivot


def load_baseline_results(baseline_dir="baseline_version"):
    base = Path(baseline_dir)
    if not base.exists():
        base = Path(__file__).resolve().parent / baseline_dir

    rows = []
    for path in base.glob("baseline_*.txt"):
        current = {"source_file": path.name}
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("Modello Backbone:"):
                current["Model"] = line.split(":", 1)[1].strip()
            elif line.startswith("Dataset:"):
                current["Dataset"] = line.split(":", 1)[1].strip()
            elif line.startswith("Learning Rate:"):
                value = line.split(":", 1)[1].strip()
                current["Learning_Rate"] = float(value)
            elif re.match(r"^micro avg\s+", line):
                parts = line.split()
                rows.append(
                    {
                        **current,
                        "Method": "full_finetuning",
                        "Micro Avg F1-Score": float(parts[-2]),
                    }
                )
                current = {"source_file": path.name}

    return pd.DataFrame(rows)


def compare_peft_with_baseline(results_dir="results/final", baseline_dir="baseline_version"):
    peft = load_peft_results(results_dir)
    baseline = load_baseline_results(baseline_dir)
    if baseline.empty:
        print("Nessun risultato baseline trovato nei log.")
        return peft, baseline, pd.DataFrame()

    peft_best = (
        peft.sort_values("Micro Avg F1-Score", ascending=False)
        .groupby(["Method", "Model", "Dataset"], as_index=False)
        .first()
    )
    combined = pd.concat(
        [
            peft_best[["Method", "Model", "Dataset", "Learning_Rate", "Micro Avg F1-Score"]],
            baseline[["Method", "Model", "Dataset", "Learning_Rate", "Micro Avg F1-Score"]],
        ],
        ignore_index=True,
    )

    pivot = combined.pivot_table(
        index=["Model", "Dataset"],
        columns="Method",
        values="Micro Avg F1-Score",
        aggfunc="max",
    )
    if "full_finetuning" in pivot.columns:
        for method in ["prefix", "soft_prompt"]:
            if method in pivot.columns:
                pivot[f"{method}_minus_full"] = pivot[method] - pivot["full_finetuning"]

    display(combined.sort_values(["Dataset", "Model", "Method"]))
    display(pivot)
    return combined, baseline, pivot


def hyperparameter_heatmaps(df, method, model_name=None, dataset_name=None):
    data = df[df["Method"] == method].copy()
    if model_name is not None:
        data = data[data["Model"] == model_name]
    if dataset_name is not None:
        data = data[data["Dataset"] == dataset_name]

    groups = list(data.groupby(["Model", "Dataset"]))
    for (model, dataset), group in groups:
        heat = group.pivot_table(
            index="Token_Length",
            columns="Learning_Rate",
            values="Micro Avg F1-Score",
        )
        plt.figure(figsize=(7, 4))
        sns.heatmap(heat, annot=True, fmt=".4f", cmap="viridis")
        plt.title(f"{method}: {model} / {dataset}")
        plt.xlabel("Learning rate")
        plt.ylabel("Prefix length")
        plt.tight_layout()
        plt.show()


def cumulative_ablation_analysis(model, loader, device, id_to_tag, file_name, drops=None):
    if drops is None:
        raise ValueError("Passa i drops restituiti da ablation_analysis oppure calcolali prima.")

    batches = list(loader)
    prefix_len = model.prefix_module.preseqlen

    drops_tensor = torch.tensor(drops, dtype=torch.float32)
    order = torch.argsort(drops_tensor, descending=True)

    with torch.no_grad():
        full_prefix = model.prefix_module(bsz=1)

    log_lines = []
    f1_values = []
    masked_positions = []

    for k in range(0, prefix_len + 1):
        ablated = full_prefix.clone().detach().contiguous()
        if k > 0:
            indices_to_mask = order[:k]
            ablated[:, indices_to_mask, :] = 0
        y_true, y_pred = collect_predictions(
            model,
            batches,
            device,
            id_to_tag,
            prefix_override=ablated,
        )
        current_f1 = f1_score(y_true, y_pred)
        f1_values.append(current_f1)
        masked_positions.append(k)

        line = f"Rimossi i {k:02d} token più importanti -> Micro F1: {current_f1:.4f}"
        print(line)
        log_lines.append(line)

    file_name = file_name.removesuffix(".pth")
    os.makedirs("output_analysis", exist_ok=True)
    with open("output_analysis/cumulative_ablation_" + file_name + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    plt.figure(figsize=(8, 4))
    plt.plot(masked_positions, f1_values, marker="o")
    plt.xlabel("Numero di prefix token rimossi, dal piu importante")
    plt.ylabel("Micro F1")
    plt.title("Ablation cumulativa dei prefix token")
    plt.tight_layout()
    plt.savefig("output_analysis/cumulative_ablation_" + file_name + ".png", dpi=120)
    plt.show()

    return pd.DataFrame({"removed_tokens": masked_positions, "f1": f1_values})

def cluster_ablation_analysis(model, val_loader, device, id_to_tag, cluster_labels, file_name=""):
    """
    Ablazione a livello di macro-cluster. Zero-out di interi gruppi funzionali
    di soft token per misurare l'impatto globale e la specializzazione dei cluster.
    """
    model.eval()
    print("Converting DataLoader to batch list (caching per velocizzare)...")
    batches = _convert_loader_to_list(val_loader)
    batch_loader = _list_to_loader(batches)
    
    # Calcolo baseline iniziale
    baseline = compute_f1(model, batch_loader, device, id_to_tag)
    print(f"Baseline F1: {baseline:.4f}\n")

    unique_clusters = np.unique(cluster_labels)
    
    with torch.no_grad():
        full_prefix = model.prefix_module(bsz=1).clone().detach().contiguous()

    log_lines = []
    log_lines.append(f"Baseline F1: {baseline:.4f}")
    log_lines.append("=== ANALISI DI ABLAZIONE PER CLUSTER ===")

    cluster_drops = []
    cluster_names = []

    for c in unique_clusters:
        # Trova gli indici dei token che appartengono al cluster corrente (0-indexed per PyTorch)
        token_indices = np.where(cluster_labels == c)[0]
        cluster_size = len(token_indices)
        
        # Crea la copia contigua e azzera l'intero blocco di token del cluster
        ablated = full_prefix.clone().detach().contiguous()
        ablated[:, token_indices, :] = 0
        
        # Ricarica il loader calcola le metriche
        batch_loader = _list_to_loader(batches)
        f1 = compute_f1(model, batch_loader, device, id_to_tag, prefix_override=ablated)
        drop = baseline - f1
        
        cluster_drops.append(drop)
        cluster_names.append(f"Cluster {c:02d}\n({cluster_size} tok)")
        
        line = f"Rimozione Cluster {c:02d} (Dimensione: {cluster_size} token) -> F1 = {f1:.4f} (drop = {drop:+.4f})"
        print(line)
        log_lines.append(line)

    # Salvataggio dei log testuali
    file_name = file_name.removesuffix(".pth")
    os.makedirs("output_analysis", exist_ok=True)
    with open("output_analysis/cluster_ablation_" + file_name + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # Generazione del grafico a barre comparativo
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(cluster_names, cluster_drops, color='crimson', edgecolor='black', alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    # Aggiunge i valori numerici sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                 f'{height:+.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel("Macro Cluster Strutturali")
    plt.ylabel("Crollo F1 (Baseline - Cluster Ablated)")
    plt.title(f"Impatto Funzionale dei Cluster di Soft Token\n({file_name})")
    plt.tight_layout()
    plt.savefig("output_analysis/cluster_ablation_" + file_name + ".png", dpi=120)
    plt.show()

    return cluster_drops

def attention_by_gold_label(model, loader, device, id_to_tag, n_batches=5):
    model.eval()
    prefix_len = model.prefix_module.preseqlen
    totals = {}
    counts = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, encoder_output = forward_with_prefix(
                model,
                input_ids,
                attention_mask,
                output_attentions=True,
            )
            attentions = torch.stack(encoder_output.attentions)
            to_prefix = attentions[:, :, :, prefix_len:, :prefix_len].mean(dim=(0, 2))

            for b in range(labels.shape[0]):
                valid_positions = labels[b] != -100
                for pos in torch.where(valid_positions)[0]:
                    label = id_to_tag[labels[b, pos].item()]
                    vec = to_prefix[b, pos].detach().cpu()
                    totals[label] = totals.get(label, torch.zeros(prefix_len)) + vec
                    counts[label] = counts.get(label, 0) + 1

    rows = []
    for label, total in totals.items():
        avg = total / max(counts[label], 1)
        rows.append({"label": label, "attention_to_prefix": avg.mean().item(), "count": counts[label]})

    df = pd.DataFrame(rows).sort_values("attention_to_prefix", ascending=False)
    display(df)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="label", y="attention_to_prefix")
    plt.xticks(rotation=45, ha="right")
    plt.title("Attenzione media verso il prefix per gold label")
    plt.tight_layout()
    plt.show()
    return df
