import dataclasses
import random
import socket
from typing import Callable, List, Tuple

import torch


def find_free_port() -> int:
    """Find a free socket to use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a port that is free
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # Return the port number


def trim_list(l: List, size: int) -> List:
    """Ensure that a list has at most size elements."""
    if len(l) <= size:
        return l
    else:
        return l[:size]


@dataclasses.dataclass
class TrainingCollation:
    x_str: List[str]
    y_str: List[str]
    all_y_options: List[List[str]]
    x_tok: List[int]
    y_tok: List[int]
    all_tokens: torch.Tensor
    loss_map: torch.Tensor
    x_introduction: List[int]
    y_introduction: List[int]


# mod for ner
class CollateForTraining:
    def __init__(self, tokenizer, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        tokens_list = [item['tokens'] for item in batch]
        ner_tags_list = [item['ner_tags'] for item in batch]

        tokenized_inputs = self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        
        labels = []
        for i, ner_tags in enumerate(ner_tags_list):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # -100 per calcolo loss
                elif word_idx != previous_word_idx:
                    label_ids.append(ner_tags[word_idx])
                else:
                    label_ids.append(ner_tags[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        return tokenized_inputs
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pretty_print_examples(
        test_prompts: List[str],
        test_responses: List[str],
        test_targets: List[List[str]],
        n_to_print: int = 5,
):
    for i in range(n_to_print):
        if len(test_targets[i]) == 1:
            print(f"{i+1}. {bcolors.UNDERLINE}{test_prompts[i]}{bcolors.END}")
            print(f'   {bcolors.BOLD}> RESPONSE:{bcolors.END} "{test_responses[i]}"')
            print(f'   {bcolors.BOLD}> EXPECTED:{bcolors.END} "{test_targets[i][0]}"')
        else:
            print(f"{i+1}. {bcolors.UNDERLINE}{test_prompts[i]}{bcolors.END}")
            print(f'   {bcolors.BOLD}> RESPONSE:{bcolors.END}   "{test_responses[i]}"')
            print(f'   {bcolors.BOLD}> EXPECTED:{bcolors.END} - "{test_targets[i][0]}"')    
            for j in range(1,len(test_targets[i])):
                print(f'               - "{test_targets[i][j]}"')    