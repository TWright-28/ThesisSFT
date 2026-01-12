from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

print("="*60)
print("FAST TEST RUN - Using 1000 examples only")
print("="*60)

print("\n" + "="*60)
print("STEP 1: Loading model and tokenizer...")
print("="*60)

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from transformers import BitsAndBytesConfig

# Load in 8-bit to save memory
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
# Reduce memory usage
model.config.use_cache = False


# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

print(f" Model loaded on: {model.device}")

print("\n" + "="*60)
print("STEP 2: Adding LoRA adapters...")
print("="*60)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n" + "="*60)
print("STEP 3: Loading data (1000 examples for fast test)...")
print("="*60)

dataset = load_dataset("json", data_files="training_dataset_masked.jsonl", split="train")

# Use only 1000 examples for fast test
dataset = dataset.select(range(min(1000, len(dataset))))
print(f" Using {len(dataset)} examples for fast test")

# Format data
def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print(" Data formatted and tokenized")

print("\n" + "="*60)
print("STEP 4: Training...")
print("="*60)

training_args = TrainingArguments(
    output_dir="./outputs_fast_test",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_grad_norm=0.3,
    learning_rate=2e-4,              # ADD THIS
    fp16=False,                       # ADD THIS (8-bit doesn't work with fp16)
    logging_steps=25,                 # ADD THIS
    save_strategy="epoch",            # ADD THIS
    save_total_limit=1,               # ADD THIS
    report_to="none",                 # ADD THIS
    gradient_checkpointing=True,      # ADD THIS
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        'input_ids': torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['input_ids']) for f in data],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['attention_mask']) for f in data],
            batch_first=True,
            padding_value=0
        ),
        'labels': torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['input_ids']) for f in data],
            batch_first=True,
            padding_value=-100
        )
    }
)

print("Expected time: ~10-20 minutes for 1000 examples")
trainer.train()

print("\n" + "="*60)
print("Saving model...")
print("="*60)

model.save_pretrained("./bug_classifier_fast_test")
tokenizer.save_pretrained("./bug_classifier_fast_test")

print(" TRAINING COMPLETE!")