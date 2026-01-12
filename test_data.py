import json
from pathlib import Path
from collections import defaultdict

def format_test_set(ground_truth_path, output_path):
    """
    Format 377 ground truth annotations as test set
    Uses ONLY title + body (matching production scenario)
    """
    print("Loading ground truth dataset...")
    ground_truth_data = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            ground_truth_data.append(json.loads(line))
    
    print(f"Loaded {len(ground_truth_data)} ground truth records")
    
    # Format for testing
    test_examples = []
    skipped = 0
    
    for record in ground_truth_data:
        # Skip if missing essential fields
        if not record.get('title') or not record.get('final_classification'):
            skipped += 1
            print(f"Warning: Skipping issue #{record.get('number', '?')} - missing title or classification")
            continue
        
        # Get body (handle None/empty)
        body = record.get('body', '').strip()
        if not body:
            body = "[No description provided]"
        
        # Truncate very long bodies
        if len(body) > 4000:
            body = body[:4000] + "\n\n[...truncated for length]"
        
        # Format the input (ONLY title + body - matching mask level 0)
        user_message = f"""Title: {record['title']}

Body:
{body}"""
        
        # Create test example in chat format
        test_example = {
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
                    "content": record['final_classification']  # Ground truth label
                }
            ],
            # Metadata for evaluation
            "metadata": {
                "issue_number": record['number'],
                "project": f"{record['owner']}/{record['repo']}",
                "ground_truth_label": record['final_classification'],
                "comments_count": record.get('comments_count', 0),
                "state": record.get('state'),
            }
        }
        
        test_examples.append(test_example)
    
    # Save test dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n{'='*60}")
    print(f" TEST DATASET CREATED")
    print(f"{'='*60}")
    print(f"Total examples: {len(test_examples)}")
    print(f"Skipped (missing data): {skipped}")
    print(f"Saved to: {output_path}")
    
    # Statistics
    label_counts = defaultdict(int)
    project_counts = defaultdict(int)
    
    for example in test_examples:
        label = example['metadata']['ground_truth_label']
        project = example['metadata']['project']
        label_counts[label] += 1
        project_counts[project] += 1
    
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH LABEL DISTRIBUTION")
    print(f"{'='*60}")
    for label in ['Intrinsic', 'Extrinsic', 'Not a Bug', 'Unknown']:
        if label in label_counts:
            count = label_counts[label]
            pct = (count / len(test_examples)) * 100
            print(f"{label:15s}: {count:5d} ({pct:5.1f}%)")
    
    # Top projects in test set
    print(f"\n{'='*60}")
    print(f"TOP 10 PROJECTS IN TEST SET")
    print(f"{'='*60}")
    for project, count in sorted(project_counts.items(), key=lambda x: -x[1])[:10]:
        pct = (count / len(test_examples)) * 100
        print(f"{project:40s}: {count:4d} ({pct:5.1f}%)")
    
    return test_examples


if __name__ == "__main__":
    print("="*60)
    print("CREATING TEST DATASET (377 GROUND TRUTH)")
    print("="*60)
    
    test_examples = format_test_set(
        ground_truth_path="issues_377.jsonl",
        output_path="test_dataset_377.jsonl"
    )
    
    # Show first example
    print(f"\n{'='*60}")
    print(f"EXAMPLE TEST RECORD")
    print(f"{'='*60}")
    example = test_examples[0]
    print(f"Project: {example['metadata']['project']} #{example['metadata']['issue_number']}")
    print(f"Ground Truth Label: {example['metadata']['ground_truth_label']}")
    print(f"\nUser Message Preview (first 300 chars):\n{example['messages'][1]['content'][:300]}...")
    print(f"\nExpected Response: {example['messages'][2]['content']}")