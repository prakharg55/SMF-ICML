from datasets import load_dataset

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def load_and_process_dataset(tokenizer, sample_size=20000):
    # Load and filter OpenAssistant
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    # Keep only high-quality English assistant responses
    filtered = dataset.filter(
        lambda x: (
            x['lang'] == 'en' and 
            x['role'] == 'assistant' and 
            x['rank'] == 0.0 or 
            x['rank'] == 1.0 and
            len(x['text']) > 50  # Filter out very short responses
        )
    )

    print(f"Filtered dataset size: {len(filtered)}")

    # Take subset
    dataset = filtered.select(range(min(sample_size, len(filtered))))

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=4  # Speed up with multiprocessing
    )

    print(f"Tokenized dataset: {tokenized}")
    return tokenized

def load_hellaswag_dataset(tokenizer, sample_size=20000):
    dataset = load_dataset("hellaswag", split="train")
    dataset = dataset.select(range(min(sample_size, len(dataset))))

    def tokenize(examples):
        return tokenizer(
            examples['ctx_a'],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=4
    )

    print(f"HellaSwag tokenized dataset: {tokenized}")
    return tokenized

def load_wikitext_dataset(tokenizer, sample_size=20000):
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    
    # Take subset from dataset
    dataset = dataset.select(range(min(sample_size, len(dataset))))

    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=4
    )

    print(f"WikiText tokenized dataset: {tokenized}")
    return tokenized
