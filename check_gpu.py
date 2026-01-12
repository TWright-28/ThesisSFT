from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

print("="*60)
print("STEP 1: Loading model and tokenizer...")
print("="*60)

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f" Model loaded: {model_name}")
print(f" Model device: {next(model.parameters()).device}")
print(f" Model dtype: {next(model.parameters()).dtype}")

print("\n" + "="*60)
print("STEP 2: Loading training data...")
print("="*60)

dataset = load_dataset("json", data_files="training_dataset_masked.jsonl", split="train")
print(f" Training examples: {len(dataset)}")

print("\n" + "="*60)
print("SUCCESS! Everything is working.")
print("="*60)