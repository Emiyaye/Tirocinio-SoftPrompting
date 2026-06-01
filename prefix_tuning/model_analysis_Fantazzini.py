import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import argparse
from trainer import prepare_data
from soft_ner_model import NERSoftPromptModel
from sklearn.decomposition import PCA
from seqeval.metrics import f1_score, classification_report
 
 
# Helprs functions
 
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
    """
    Convert a DataLoader to a list of batches to avoid deadlock issues
    with num_workers > 0 when iterating multiple times in a loop.
    This is necessary because repeatedly iterating through a DataLoader
    with num_workers > 0 can cause worker processes to deadlock.
    """
    batches = []
    for batch in loader:
        batches.append(batch)
    return batches


def _list_to_loader(batches):
    """
    Wrap a list of batches as an iterable that behaves like a DataLoader.
    """
    class BatchIterator:
        def __init__(self, batch_list):
            self.batches = batch_list
        
        def __iter__(self):
            return iter(self.batches)
    
    return BatchIterator(batches)


#####################
# Position ablation #
#####################

# Remove one soft token at the time to better measure the F1 chages
# This way we can see which prefix positions are most important for the model's performance

def ablation_analysis(model, val_loader, device, id_to_tag, file_name = ""):
    """Zero out one prefix position at a time, measure F1 drop."""
    
    # Convert the DataLoader to a list of batches to prevent deadlocks
    # when calling compute_f1 multiple times in the loop below
    print("Converting DataLoader to batch list (this may take a moment)...")
    batches = _convert_loader_to_list(val_loader)
    batch_loader = _list_to_loader(batches)
    
    baseline = compute_f1(model, batch_loader, device, id_to_tag)
    print(f"Baseline F1: {baseline:.4f}")

    prefix_len = model.prefix_module.preseqlen
    with torch.no_grad():
        full_prefix = model.prefix_module(bsz=1)     # (1, L, H)

    drops = []
    for i in range(prefix_len):
        ablated = full_prefix.clone()
        ablated[:, i, :] = 0
        # Recreate the batch loader for each iteration to avoid iterator exhaustion
        batch_loader = _list_to_loader(batches)
        f1 = compute_f1(model, batch_loader, device, id_to_tag,
                        prefix_override=ablated)
        drops.append(baseline - f1)
        print(f"  position {i:02d}: F1 = {f1:.4f}  (drop = {baseline - f1:+.4f})")

    plt.figure(figsize=(12, 4))
    plt.bar(range(prefix_len), drops)
    plt.xlabel("Prefix position"); plt.ylabel("F1 drop when ablated")
    plt.title("Importance of each prefix position")
    file_name = file_name.removesuffix(".pth")
    plt.tight_layout(); plt.savefig("ablation_" + file_name + ".png", dpi=120); plt.show()
    return drops
 
 
###########################
# Attention to the prefix #                                                
###########################
 
# Check how the attention interacts wtich the soft tokens
 
def attention_analysis(model, val_loader, device, n_batches=5, file_name=""):
    """How much do real tokens attend to each prefix position?"""
    model.eval()
    prefix_len = model.prefix_module.preseqlen
    total = torch.zeros(prefix_len)
    n = 0
 
    with torch.no_grad():
        for b, batch in enumerate(val_loader):
            if b >= n_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attn      = batch['attention_mask'].to(device)
 
            _, enc_out = forward_custom(model, input_ids, attn,
                                        output_attentions=True)
            # enc_out.attentions: tuple of L tensors, each (B, heads, seq, seq)
            a = torch.stack(enc_out.attentions)               # (L, B, H, S, S)
            # rows = real tokens, cols = prefix tokens
            to_prefix = a[:, :, :, prefix_len:, :prefix_len]  # (L, B, H, real, pref)
            total += to_prefix.mean(dim=(0, 1, 2, 3)).cpu()
            n += 1
 
    avg = total / n
    plt.figure(figsize=(12, 4))
    plt.bar(range(prefix_len), avg.numpy())
    plt.xlabel("Prefix position"); plt.ylabel("Avg attention received")
    plt.title("Which prefix positions do real tokens attend to?")
    file_name = file_name.removesuffix(".pth")
    plt.tight_layout(); plt.savefig("attention_" + file_name +".png", dpi=120); plt.show()
    return avg
 
 
 
#######
# PCA #    
#######
 
# Where do the prefix embeddings live relative to vocabulary?  
# Should be a better analysis of the soft tokens positioning wrt to the backbone's vocabulary  
 
def pca_analysis(model, n_vocab=5000, seed=0, file_name=""):
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        prefix = model.prefix_module(bsz=1).squeeze(0).cpu().numpy()          # (L, H)
        vocab  = model.encoder.get_input_embeddings().weight.cpu().numpy()    # (V, H)
 
    idx = rng.choice(vocab.shape[0], size=min(n_vocab, vocab.shape[0]),
                     replace=False)
    vocab_s = vocab[idx]
 
    proj = PCA(n_components=2).fit_transform(np.vstack([vocab_s, prefix]))
    v2d, p2d = proj[:len(vocab_s)], proj[len(vocab_s):]
 
    plt.figure(figsize=(8, 7))
    plt.scatter(v2d[:, 0], v2d[:, 1], alpha=0.25, s=6, label="Vocabulary")
    plt.scatter(p2d[:, 0], p2d[:, 1], color="red", s=60, marker="x",
                label="Prefix tokens")
    for i, (x, y) in enumerate(p2d):
        plt.annotate(str(i), (x, y), fontsize=7)
    plt.legend(); plt.title("Prefix embeddings vs vocabulary (PCA)")
    file_name = file_name.removesuffix(".pth")
    plt.tight_layout(); plt.savefig("pca_" + file_name + ".png" , dpi=120); plt.show()
 
 
 
 
# Example
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Parametri Modello e Dataset
    parser.add_argument('--model', type=str, default='dmis-lab/biobert-v1.1')
    parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr')
    
    # Iperparametri, Devono coincidere con quelli usati durante l'addestramento
    # batch_size 8 bc5cdr, 16 anatem jnlpba
    parser.add_argument('--prefix_length', type=int, default=50)
    parser.add_argument('--mid_dim', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Percorso del file .pth caricare. Se None, lo costruisce automaticamente.')

    args = parser.parse_args()
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    device = torch.device(args.device)

    if args.model_path is None:
        clean_model_name = args.model.split('/')[-1]
        clean_dataset_name = args.dataset.split('/')[-1]
        file_name = f"model_{clean_model_name}_{clean_dataset_name}_len{args.prefix_length}_lr{args.learning_rate}.pth"
    else:
        file_name = args.model_path


    dataloaders, ner_tags, _ = prepare_data(dataset_name=args.dataset, model_name= args.model, max_seq_len=args.max_seq_len, subset_size=-1, batch_size=args.batch_size, device=device)
    id_to_tag = {i: tag for i, tag in enumerate(ner_tags)}
    model = NERSoftPromptModel(model_name=args.model, ner_tags=ner_tags, prefix_length=args.prefix_length).to(device)
    model.load_state_dict(torch.load(file_name, map_location=device))
    if hasattr(model.encoder, "config"):
        model.encoder.config._attn_implementation = "eager"

    val_loader = dataloaders['test']   # or 'validation'
   
    pca_analysis(model, file_name = file_name)
    attention_analysis(model, val_loader, device, file_name= file_name)
    #ablation_analysis(model, val_loader, device, id_to_tag, file_name= file_name)
 