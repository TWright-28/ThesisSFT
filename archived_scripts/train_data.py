import json
import random
from pathlib import Path
from collections import defaultdict

def format_issue_with_metadata(record, mask_level):
    """
    Format issue with different levels of metadata masking
    
    mask_level:
        0 = Title + Body only (50% of training)
        1 = Title + Body + First comment (25% of training)
        2 = Title + Body + All comments (15% of training)
        3 = Full context: Title + Body + Comments + State + Labels (10% of training)
    """
    # Always include title and body
    body = record.get('body', '').strip()
    if not body:
        body = "[No description provided]"
    
    # Truncate very long bodies
    if len(body) > 4000:
        body = body[:4000] + "\n\n[...truncated for length]"
    
    parts = [f"Title: {record['title']}", f"\nBody:\n{body}"]
    
    # Mask level 0: Just title + body
    if mask_level == 0:
        return "\n".join(parts)
    
    # Mask level 1+: Add comments
    if mask_level >= 1 and record.get('comments') and len(record['comments']) > 0:
        comments_to_include = record['comments']
        
        # Mask level 1: Only first comment
        if mask_level == 1:
            comments_to_include = comments_to_include[:1]
        
        # Format comments
        comments_text = "\n\nComments:\n"
        for comment in comments_to_include:
            author = comment['author']['username']
            assoc = comment['author'].get('author_association', 'NONE')
            body = comment['body'][:500]  # Truncate long comments
            comments_text += f"[{assoc}] {author}: {body}\n"
        
        parts.append(comments_text)
    
    # Mask level 3: Add state and labels
    if mask_level >= 3:
        parts.append(f"\nState: {record.get('state', 'open')}")
        
        labels = record.get('labels', [])
        if labels:
            label_names = [l.get('name', l) if isinstance(l, dict) else l for l in labels]
            parts.append(f"Labels: {', '.join(label_names)}")
        else:
            parts.append("Labels: []")
        
        # Add closing PR info if exists
        closing_pr = record.get('closing_pr')
        if closing_pr:
            parts.append(f"Closing PR: #{closing_pr.get('number', 'unknown')}")
        else:
            parts.append("Closing PR: None")
    
    return "\n".join(parts)


def format_for_training(merged_path, output_path, include_reasoning=False, 
                       masking_distribution=None):
    """
    Convert merged data into SFT training format with progressive masking
    
    Args:
        merged_path: Path to merged dataset
        output_path: Path to save training dataset
        include_reasoning: Include Gemini's reasoning in output
        masking_distribution: Dict of mask_level -> probability
                             Default: {0: 0.50, 1: 0.25, 2: 0.15, 3: 0.10}
    """
    if masking_distribution is None:
        masking_distribution = {
            0: 0.50,  # 50% title+body only
            1: 0.25,  # 25% + first comment
            2: 0.15,  # 15% + all comments
            3: 0.10,  # 10% full context
        }
    
    print("Loading merged dataset...")
    merged_data = []
    with open(merged_path, 'r', encoding='utf-8') as f:
        for line in f:
            merged_data.append(json.loads(line))
    
    print(f"Loaded {len(merged_data)} records")
    print(f"\nMasking distribution:")
    for level, prob in sorted(masking_distribution.items()):
        desc = {
            0: "Title + Body only",
            1: "Title + Body + First comment",
            2: "Title + Body + All comments",
            3: "Full context (+ State + Labels)"
        }
        print(f"  Level {level} ({desc[level]}): {prob*100:.0f}%")
    
    # Format for training
    training_examples = []
    skipped = 0
    mask_level_counts = defaultdict(int)
    
    random.seed(42)  # For reproducibility
    
    for record in merged_data:
        # Skip if missing essential fields
        if not record.get('title') or not record.get('predicted_label'):
            skipped += 1
            continue
        
        # Randomly select mask level based on distribution
        rand = random.random()
        cumulative = 0
        mask_level = 0
        for level, prob in sorted(masking_distribution.items()):
            cumulative += prob
            if rand < cumulative:
                mask_level = level
                break
        
        mask_level_counts[mask_level] += 1
        
        # Format the input with selected mask level
        user_message = format_issue_with_metadata(record, mask_level)
        
        # Format the output
        if include_reasoning and record.get('reasoning'):
            # Include Gemini's reasoning (chain-of-thought)
            assistant_message = f"{record['reasoning']}\n\nFinal Answer: {record['predicted_label']}"
        else:
            # Just the label
            assistant_message = record['predicted_label']
        
        # Create training example in chat format
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bug classification expert. Classify GitHub issues into one of four categories: Intrinsic (bugs in the code itself), Extrinsic (bugs from external dependencies or environment), Not a Bug (user errors, feature requests, questions), or Unknown (insufficient information)."
                },
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ],
            # Metadata for tracking (not used in training)
            "metadata": {
                "issue_number": record['issue_number'],
                "project": record['project'],
                "predicted_label": record['predicted_label'],
                "confidence_score": record.get('confidence_score', 0.0),
                "comments_count": record.get('comments_count', 0),
                "mask_level": mask_level,
            }
        }
        
        training_examples.append(training_example)
    
    # Save training dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n{'='*60}")
    print(f" TRAINING DATASET CREATED")
    print(f"{'='*60}")
    print(f"Total examples: {len(training_examples)}")
    print(f"Skipped (missing data): {skipped}")
    print(f"Saved to: {output_path}")
    
    # Masking statistics
    print(f"\n{'='*60}")
    print(f"ACTUAL MASK LEVEL DISTRIBUTION")
    print(f"{'='*60}")
    for level in sorted(mask_level_counts.keys()):
        count = mask_level_counts[level]
        pct = (count / len(training_examples)) * 100
        desc = {
            0: "Title + Body only",
            1: "+ First comment",
            2: "+ All comments",
            3: "+ Full context"
        }
        print(f"Level {level} ({desc[level]:20s}): {count:5d} ({pct:5.1f}%)")
    
    # Label statistics
    label_counts = defaultdict(int)
    confidence_by_label = defaultdict(list)
    
    for example in training_examples:
        label = example['metadata']['predicted_label']
        conf = example['metadata'].get('confidence_score', 0.0)  # FIX: Handle None
        
        label_counts[label] += 1
        if conf is not None and conf > 0:  # FIX: Check for None first
            confidence_by_label[label].append(conf)
    
    print(f"\n{'='*60}")
    print(f"LABEL DISTRIBUTION")
    print(f"{'='*60}")
    for label in ['Intrinsic', 'Extrinsic', 'Not a Bug', 'Unknown']:
        if label in label_counts:
            count = label_counts[label]
            pct = (count / len(training_examples)) * 100
            avg_conf = sum(confidence_by_label[label]) / len(confidence_by_label[label]) if confidence_by_label[label] else 0
            print(f"{label:15s}: {count:5d} ({pct:5.1f}%) - Avg conf: {avg_conf:.3f}")
    
    return training_examples


if __name__ == "__main__":
    print("="*60)
    print("CREATING TRAINING DATASET WITH PROGRESSIVE MASKING")
    print("="*60)
    
    # Create training dataset with progressive masking
    training_examples = format_for_training(
        merged_path="merged_7k_training_data.jsonl",
        output_path="training_dataset_masked.jsonl",
        include_reasoning=False,  # Just labels for now
        masking_distribution={
            0: 0.50,  # 50% title+body only (simulates new issues)
            1: 0.25,  # 25% + first comment
            2: 0.15,  # 15% + all comments
            3: 0.10,  # 10% full context
        }
    )
    
    # Show examples of each mask level
    print(f"\n{'='*60}")
    print(f"EXAMPLE TRAINING RECORDS (ONE PER MASK LEVEL)")
    print(f"{'='*60}")
    
    for level in [0, 1, 2, 3]:
        examples_at_level = [ex for ex in training_examples if ex['metadata']['mask_level'] == level]
        if examples_at_level:
            print(f"\n--- MASK LEVEL {level} EXAMPLE ---")
            example = examples_at_level[0]
            print(f"Project: {example['metadata']['project']} #{example['metadata']['issue_number']}")
            print(f"Label: {example['metadata']['predicted_label']}")
            print(f"\nUser Message Preview (first 300 chars):\n{example['messages'][1]['content'][:300]}...")
            print(f"\nAssistant Response: {example['messages'][2]['content']}")