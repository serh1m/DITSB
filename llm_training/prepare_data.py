import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def prepare_data(args):
    print(f"Loading dataset: {args.dataset}")
    # We use a reliable default dataset if openwebtext is too large to test.
    # Users can pass --dataset "Salesforce/wikitext" or any valid HF dataset.
    try:
        if args.dataset == 'wikitext' or "tiny_shakespeare" in args.dataset:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        else:
            dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        print(f"Error loading {args.dataset}, falling back to stable wikitext-2-raw-v1: {e}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    # Using a typical BPE tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Do not pad or truncate here; just convert text to tokens
        return tokenizer(examples["text"], truncation=False)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder, making sure each sequence is precisely `args.seq_len` long with zero padding!
        total_length = (total_length // args.seq_len) * args.seq_len
        
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Packing token sequences continuously (Zero Padding)...")
    packed_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=8)

    print("Formatting and saving to local memmap...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Using memmap for fast streaming large data slices directly into DataLoader
    all_input_ids = []
    for item in packed_datasets:
        all_input_ids.append(item['input_ids'])
        
    arr = np.array(all_input_ids, dtype=np.uint16)
    np.save(os.path.join(args.output_dir, "input_ids.npy"), arr)

    print(f"Data prepared successfully at {args.output_dir}. Total sequences: {len(arr)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2") # typically a real BPE
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./data/openwebtext_processed")
    args = parser.parse_args()
    prepare_data(args)
