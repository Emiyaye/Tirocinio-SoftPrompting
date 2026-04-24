import torch
import sys
import io
import argparse
from trainer import train_ner_prefix_tuning_model, evaluate_model, eprint, prepare_data, interpret_soft_tokens
from soft_ner_model import NERSoftPromptModel

def train(args, device, out_file_name):
    ## Training
    train_losses, val_losses, saved_path = train_ner_prefix_tuning_model(
        model_name=args.model,
        dataset_name=args.dataset,
        prefix_length=args.prefix_length,
        mid_dim=args.mid_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_len=args.max_seq_len,
        patience=args.patience,
        subset_size=args.subset_size,
        file_name=out_file_name,
        device=device
    )

    print("\nTraining concluso. Risultati per epoca:")
    for i, (t_l, v_l) in enumerate(zip(train_losses, val_losses)):
        print(f"Epoch {i+1:02d}: Train Loss {t_l:.4f} | Val Loss {v_l:.4f}")
        
def test(args, device, in_file_name):
    # Test
    print("\nInizio valutazione finale sul set di test")
    evaluate_model(
        model_name=args.model,
        dataset_name=args.dataset,
        file_name=in_file_name,
        prefix_length=args.prefix_length,
        mid_dim=args.mid_dim,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device
    )
    

def reverse_embedding(args, device, in_file_name):
    
    print("\nAVVIO ANALISI REVERSE EMBEDDING")
    dataloaders, ner_tags, tokenizer = prepare_data(dataset_name=args.dataset, model_name=args.model, max_seq_len=args.max_seq_len, subset_size=10, batch_size=args.batch_size, device=device)
    if "roberta-base-uncased" in args.model.lower():
        args.model = "roberta-base"
    model = NERSoftPromptModel(model_name=args.model, ner_tags=ner_tags, prefix_length=args.prefix_length, mid_dim=args.mid_dim).to(device)
    model.load_state_dict(torch.load(in_file_name, map_location=device))
    
    interpret_soft_tokens(model, tokenizer, in_file_name, k=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Parametri Modello e Dataset
    parser.add_argument('--model', type=str, default='dmis-lab/biobert-v1.1',
                        help='Percorso o nome del modello Transformer su Hugging Face (default: BioBERT v1.1).')
    parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr',
                        help='Nome del dataset NER da caricare (default: BC5CDR).')
    
    # Iperparametri Prefix Tuning
    parser.add_argument('--prefix_length', type=int, default=50,
                        help='Numero di token virtuali da aggiungere all\'inizio della sequenza.')
    parser.add_argument('--mid_dim', type=int, default=512,
                        help='Dimensione della proiezione intermedia per l\'MLP del prefisso.')

    # Parametri di Addestramento
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Numero di esempi processati per ogni step (ridurre se finisce la memoria GPU).')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Tasso di apprendimento per i parametri addestrabili.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Numero massimo di epoche di addestramento.')
    parser.add_argument('--patience', type=int, default=6,
                        help='Numero di epoche di attesa senza miglioramenti prima di interrompere (Early Stopping).')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Lunghezza massima dei token in input (testo + prefisso).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Dispositivo di calcolo da utilizzare (cuda o cpu).')
    parser.add_argument('--subset_size', type=int, default=-1,
                        help='Numero di esempi da usare per addestramento e validazione (-1 per l\'intero dataset).')
    
    args = parser.parse_args()
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Generazione automatica di un nome file unico
    clean_model_name = args.model.split('/')[-1]
    clean_dataset_name = args.dataset.split('/')[-1]
    file_save_name = f"model_{clean_model_name}_{clean_dataset_name}_len{args.prefix_length}_lr{args.learning_rate}.pth"

    device = torch.device(args.device)

    print(f"CONFIGURAZIONE ESPERIMENTO")
    print(f"Modello:  {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Prefix:   {args.prefix_length} token")
    print(f"LR:       {args.learning_rate}")
    print(f"Output:   {file_save_name}")
    print()
    
    train(args, device, file_save_name)
    
    test(args, device, file_save_name)
    
    reverse_embedding(args, device, file_save_name)
    
    