from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

print("="*60)
print("STEP 1: Loading model with Unsloth...")
print("="*60)

max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True  # Use 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(f"✓ Model loaded successfully")

print("\n" + "="*60)
print("STEP 2: Adding LoRA adapters...")
print("="*60)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (can increase to 32 for better quality)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("✓ LoRA adapters added")

print("\n" + "="*60)
print("STEP 3: Loading training data...")
print("="*60)

dataset = load_dataset("json", data_files="training_dataset_masked.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} training examples")

# Format chat data for training
def formatting_prompts_func(examples):
    """Convert chat messages to the Llama format"""
    convos = examples["messages"]
    texts = []
    
    for convo in convos:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

# Apply formatting
dataset = dataset.map(
    formatting_prompts_func, 
    batched=True,
    remove_columns=dataset.column_names
)

print("✓ Data formatted for training")
print(f"  Example length: {len(dataset[0]['text'])} characters")

print("\n" + "="*60)
print("STEP 4: Setting up trainer...")
print("="*60)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,  # Adjust as needed
        # max_steps = 500,  # Or use max_steps instead
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 2,  # Keep only 2 checkpoints
    ),
)

print("✓ Trainer configured")
print(f"  Batch size: 2")
print(f"  Gradient accumulation: 4 (effective batch size: 8)")
print(f"  Total steps: ~{len(dataset) // 8} per epoch")

print("\n" + "="*60)
print("STEP 5: Starting training...")
print("="*60)

trainer_stats = trainer.train()

print("\n" + "="*60)
print("STEP 6: Saving model...")
print("="*60)

# Save LoRA adapters
model.save_pretrained("bug_classifier_lora")
tokenizer.save_pretrained("bug_classifier_lora")

print("✓ Training complete!")
print(f"✓ LoRA adapters saved to: bug_classifier_lora/")
print(f"\nTraining stats:")
print(f"  Total steps: {trainer_stats.global_step}")
print(f"  Training loss: {trainer_stats.training_loss:.4f}")

# Optional: Save merged model (full model + LoRA)
print("\n" + "="*60)
print("STEP 7: Saving merged model (optional)...")
print("="*60)

model.save_pretrained_merged("bug_classifier_merged", tokenizer, save_method="merged_16bit")
print("✓ Merged model saved to: bug_classifier_merged/")