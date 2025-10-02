import pandas as pd
import re

def parse_training_logs(file_path):
    """
    Parses a text file containing training logs and extracts data into a structured format.
    """
    data = []
    current_model = None
    current_dataset = None
    current_token_len = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # new model/dataset/token_lenght
        model_match = re.search(r'--model "([^"]+)" --dataset "([^"]+)" --token_lenght "([^"]+)"', line)
        if model_match:
            current_model = model_match.group(1)
            current_dataset = model_match.group(2)
            current_token_len = int(model_match.group(3))
            continue

        log_match = re.search(r'Epoca (\d+)/(\d+), Perdita \(Loss\): ([\d.]+), Perdita di Validazione \(Loss\): ([\d.]+)', line)
        if log_match and current_model and current_dataset and current_token_len is not None:
            epoch = int(log_match.group(1))
            total_epochs = int(log_match.group(2))
            loss = float(log_match.group(3))
            validation_loss = float(log_match.group(4))

            data.append({
                'Model': current_model,
                'Dataset': current_dataset,
                'Token Length': current_token_len,
                'Epoch': epoch,
                'Training Loss': loss,
                'Validation Loss': validation_loss
            })

    return pd.DataFrame(data)

def parse_testing_logs(file_path):
    """
    Parses a text file containing testing classification reports
    """
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()

    # delimiter
    blocks = re.split(r'model = (.+)', content)
    
    for i in range(1, len(blocks), 2):
        config_line = blocks[i]
        report_text = blocks[i+1]
        
        config_match = re.search(r'([^,]+), dataset = ([^,]+), token lenght = (\d+)', config_line.strip())
        if not config_match:
            continue
            
        current_model = config_match.group(1).strip()
        current_dataset = config_match.group(2).strip()
        current_token_len = int(config_match.group(3))

        for line in report_text.split('\n'):
            line = line.strip()

            if line and not line.startswith('precision'):
                report_match = re.search(r'(\w+ ?\w*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)', line)
                if report_match:
                    category = report_match.group(1).strip()
                    precision = float(report_match.group(2))
                    recall = float(report_match.group(3))
                    f1_score = float(report_match.group(4))
                    support = int(report_match.group(5))

                    data.append({
                        'Model': current_model,
                        'Dataset': current_dataset,
                        'Token Length': current_token_len,
                        'Category': category,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1_score,
                        'Support': support
                    })
    
    return pd.DataFrame(data)

training_log_path = 'training_results.txt'
testing_log_path = 'testing_results.txt'

training_df = parse_training_logs(training_log_path)
testing_df = parse_testing_logs(testing_log_path)

training_output_path = 'training_results.csv'
testing_output_path = 'testing_results.csv'

training_df.to_csv(training_output_path, index=False, sep=';')
testing_df.to_csv(testing_output_path, index=False, sep=';')

print(f"I dati di training sono stati salvati in '{training_output_path}'")
print(f"I dati di testing sono stati salvati in '{testing_output_path}'")