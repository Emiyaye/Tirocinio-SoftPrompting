import torch
import sys
import io
import argparse
from prefix_tuning.baseline_version.baseline_trainer import train_ner_baseline_model, evaluate_baseline_model, eprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER Baseline Full Fine-Tuning")
    
    parser.add_argument('--model', type=str, default='dmis-lab/biobert-v1.1')
    parser.add_argument('--dataset', type=str, default='disi-unibo-nlp/bc5cdr')



    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--subset_size', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = torch.device(args.device)

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    clean_model_name = args.model.split('/')[-1]
    clean_dataset_name = args.dataset.split('/')[-1]
    file_save_name = f"baseline_{clean_model_name}_{clean_dataset_name}.pth"

    print("AVVIO ESPERIMENTO: BASELINE FULL FINE-TUNING")
    print(f"Modello Backbone: {args.model}")
    print(f"Dataset:          {args.dataset}")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Output File:      {file_save_name}")
    print(f"Dispositivo:      {device}")
    print()

    train_losses, val_losses, saved_model_path = train_ner_baseline_model(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_len=args.max_seq_len,
        subset_size=args.subset_size,
        file_name=file_save_name,
        patience=args.patience,
        device=device
    )

    print("\nAddestramento terminato con successo.")
    print(f"Modello migliore salvato in: {saved_model_path}")

    print("\nValutazione delle prestazioni sul Test Set...")
    evaluate_baseline_model(
        model_name=args.model,
        dataset_name=args.dataset,
        file_name=saved_model_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device
    )