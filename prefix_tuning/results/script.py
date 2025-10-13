import pandas as pd
import re
from typing import List, Dict, Any

def parse_logs(file_path: str, trained_param: str):
    """
    parse with regex from txt into csv
    """
 
    data: List[Dict[str, Any]] = []

    current_model = None
    current_dataset = None
    trained_param_value = None

    with open(file_path, 'r') as f:
        content = f.read()

    # suddivizione in blocchi
    blocks = re.split(r'(?=\n?model = )', content)

    for block in blocks:
        if not block.strip():
            continue

        model_match = re.search(fr'model = ([^\s]+) dataset = ([^\s]+) {trained_param} = ([^\s]+)', block)
        if model_match:
            current_model = model_match.group(1)
            current_dataset = model_match.group(2)
            try:
                trained_param_value = int(model_match.group(3))
            except ValueError:
                trained_param_value = model_match.group(3)
        else:
            continue
            
        micro_avg_match = re.search(r'micro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', block)

        if micro_avg_match:
            micro_avg_f1 = float(micro_avg_match.group(1))
            
            data.append({
                'Model': current_model,
                'Dataset': current_dataset,
                trained_param: trained_param_value,
                'Micro Avg F1-Score': micro_avg_f1
            })

    return pd.DataFrame(data)

file_path = "learning_rate/testing_results.txt"
df = parse_logs(file_path, "learning_rate")

output_path = 'testing_results.csv'

df.to_csv(output_path, index=False, sep=';')
