# %%
import math
import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import NFD, StripAccents, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import BertTokenizerFast, AutoTokenizer
from tqdm import tqdm  # これを使います

jsonl_path = "/workspace/0-utils/1-data/pubmed/pmid_absttext.jsonl"
target_sizes = [30000, 50000, 100000, 150000]

special_tokens_list = ["[PAD]"] 
special_tokens_list += [f"[unused{i}]" for i in range(1, 100)]
special_tokens_list += ["[UNK]", "[CLS]", "[SEP]", "[MASK]"]

print(f"Loading dataset from {jsonl_path}...")
dataset = load_dataset("json", data_files=jsonl_path, split="train")
total_samples = len(dataset)

def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, total_samples, batch_size), desc="Feeding batches", unit="batch"):
        yield dataset[i : i + batch_size]["text"]

for raw_size in target_sizes:
    aligned_vocab_size = math.ceil(raw_size / 128) * 128
    print(f"\nTarget: {raw_size} -> Aligned: {aligned_vocab_size}")

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = BertPreTokenizer()

    trainer = trainers.WordPieceTrainer(
        vocab_size=aligned_vocab_size, 
        special_tokens=special_tokens_list, 
        show_progress=True 
    )

    print(f"Training tokenizer for size {aligned_vocab_size}...")
    
    tokenizer.train_from_iterator(
        batch_iterator(), 
        trainer=trainer, 
        length=total_samples
    )

    output_dir = f"pubmed_bert_vocab_{aligned_vocab_size}"
    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer, do_lower_case=False)
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Saved to: {output_dir}")

print("\nAll Done!")
# %%
import glob
paths = glob.glob("pubmed_bert_vocab_*/tokenizer.json")

for path in paths:
    print(f"\nLoading tokenizer from {path}...")
    tokenizer = Tokenizer.from_file(path)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    tokens = [token for token, id in sorted_vocab]

    output_vocab_file = f"{os.path.dirname(path)}/vocab.txt"
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")
    print(f"Saved vocab.txt to: {output_vocab_file}")

# %% 
def verify_tokenizer_compatibility(created_model_path, expected_vocab_size):
    print(f"\n[Verification] Checking model at: {created_model_path} ...")
    
    my_tokenizer = AutoTokenizer.from_pretrained(created_model_path)
    ref_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")

    print(f"  > Reference: {ref_tokenizer.name_or_path}")
    
    check_tokens = ["pad_token", "unk_token", "cls_token", "sep_token", "mask_token"]
    all_passed = True

    print(f"  > {'Token Name':<12} | {'My ID':<8} | {'Ref ID':<8} | {'Status'}")
    print("  " + "-"*45)

    for token_name in check_tokens:
        my_id = getattr(my_tokenizer, f"{token_name}_id")
        ref_id = getattr(ref_tokenizer, f"{token_name}_id")
        
        status = "✅ OK" if my_id == ref_id else "❌ NG"
        if my_id != ref_id:
            all_passed = False
        
        print(f"  > {token_name:<12} | {my_id:<8} | {ref_id:<8} | {status}")

    size_status = "✅ OK" if my_tokenizer.vocab_size == expected_vocab_size else "❌ NG"
    print(f"  > {'vocab_size':<12} | {my_tokenizer.vocab_size:<8} | {expected_vocab_size:<8} | {size_status}")
    
    if my_tokenizer.vocab_size != expected_vocab_size:
        all_passed = False

    if all_passed:
        print("\n  🎉 Verification Passed! This tokenizer is fully compatible with BERT ID mapping.")
    else:
        print("\n  ⚠️ Verification FAILED! Please check the output.")
    
    return all_passed

for raw_size in target_sizes:
    aligned_vocab_size = math.ceil(raw_size / 128) * 128
    output_dir = f"pubmed_bert_vocab_{aligned_vocab_size}"
    
    verify_tokenizer_compatibility(output_dir, aligned_vocab_size)
# %%
