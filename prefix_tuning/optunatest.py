import optuna
import torch
import sys
import io
import os
import shutil
import traceback
from trainer import train_ner_prefix_tuning_model, eprint

best_losses = {"anatem": float('inf'), "jnlpba": float('inf')}

def objective(trial):
    global best_overall_loss

    prefix_length = trial.suggest_categorical('prefix_length', [10, 30, 50, 80, 100])
    
    dataset_name = trial.suggest_categorical('dataset', ["disi-unibo-nlp/jnlpba", "disi-unibo-nlp/anatem"])
    model_name = "dmis-lab/biobert-v1.1" 
    
    clean_ds = dataset_name.split('/')[-1]
    print("\n" + "="*50)
    print(f" INIZIO TRIAL #{trial.number}")
    print(f" Dataset: {clean_ds} | Prefix Length: {prefix_length}")
    print("="*50 + "\n")
    
    eprint(f" INIZIO TRIAL #{trial.number}")
    eprint(f" Dataset: {clean_ds} | Prefix Length: {prefix_length}")

    temp_file = f"temp_trial_{clean_ds}_len{prefix_length}.pth"

    try:
        _, val_losses, _ = train_ner_prefix_tuning_model(
            model_name=model_name,
            dataset_name=dataset_name,
            prefix_length=prefix_length,
            mid_dim=512,
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=30,
            patience=5,
            file_name=temp_file,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        current_best_loss = min(val_losses)
        
        # save pth
        if current_best_loss < best_losses[clean_ds]:
            best_losses[clean_ds] = current_best_loss
            final_save_name = f"best_model_{clean_ds}_len{prefix_length}.pth"
            shutil.copy(temp_file, final_save_name)
            eprint(f"NUOVO RECORD: Modello salvato come {final_save_name}")
        
        # rm temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return current_best_loss

    except Exception as e:
        eprint(f"Errore nel trial: {e}")
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


    study = optuna.create_study(
        study_name="ner_prefix_optimization", 
        storage="sqlite:///optuna_ner_study.db", 
        load_if_exists=True,
        direction="minimize"
    )

    # Avvio ottimizzazione
    
    study.optimize(objective, n_trials=6)
    
    print("\n"+"="*40)
    print("RISULTATI OTTIMIZZAZIONE:")
    print(f"Miglior Dataset/Config: {study.best_params}")
    print(f"Miglior Validation Loss: {study.best_value:.4f}")
    print("="*40)