import json
from pathlib import Path
from collections import defaultdict

def load_jsonl(filepath):
    """Load JSONL file into a list of dicts"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
    return data

def normalize_project_name(owner, repo=None):
    """
    Normalize project name to handle different formats
    
    Handles:
    - "owner/repo" format (from predictions)
    - Separate owner, repo fields (from original data)
    """
    if repo is None:
        # Already in "owner/repo" format
        return owner.lower()
    else:
        # Separate fields, combine them
        return f"{owner}/{repo}".lower()

def merge_datasets(original_path, predictions_path, output_path):
    """
    Merge original bug data with Gemini predictions
    Uses composite key: (project, issue_number)
    """
    print("Loading datasets...")
    original_data = load_jsonl(original_path)
    predictions = load_jsonl(predictions_path)
    
    print(f"Original data: {len(original_data)} records")
    print(f"Predictions: {len(predictions)} records")
    
    # Create lookup dictionary with composite key (project, issue_number)
    # Original data has separate 'owner' and 'repo' fields
    orig_lookup = {}
    for bug in original_data:
        project = normalize_project_name(bug['owner'], bug['repo'])
        issue_num = bug['number']
        key = (project, issue_num)
        
        if key in orig_lookup:
            print(f"⚠️  Duplicate found in original data: {key}")
        
        orig_lookup[key] = bug
    
    print(f"Original data unique keys: {len(orig_lookup)}")
    
    # Merge
    merged = []
    matched = 0
    unmatched_predictions = []
    
    for pred in predictions:
        # Predictions have 'project' field in "owner/repo" format
        project = normalize_project_name(pred['project'])
        issue_num = pred['issue_number']
        key = (project, issue_num)
        
        if key in orig_lookup:
            orig = orig_lookup[key]
            
            # Verify the match is correct
            assert orig['number'] == issue_num, f"Issue number mismatch for {key}"
            
            # Create merged record
            merged_record = {
                # Identifiers
                'issue_number': issue_num,
                'project': pred['project'],  # Use "owner/repo" format
                'owner': orig['owner'],
                'repo': orig['repo'],
                'html_url': orig['html_url'],
                
                # Issue content
                'title': orig['title'],
                'body': orig['body'],
                'state': orig['state'],
                'state_reason': orig.get('state_reason'),
                'locked': orig.get('locked', False),
                
                # Timestamps
                'created_at': orig['created_at'],
                'updated_at': orig['updated_at'],
                'closed_at': orig.get('closed_at'),
                
                # Comments
                'comments_count': orig['comments_count'],
                'comments': orig['comments'],  # Full comment objects
                'comments_text': orig['comments_text'],  # Formatted text
                
                # Metadata
                'labels': orig['labels'],
                'author': orig['author'],
                'closed_by': orig.get('closed_by'),
                'closing_pr': orig['closing_pr'],
                'closing_commit': orig['closing_commit'],
                'assignees': orig.get('assignees', []),
                'milestone': orig.get('milestone'),
                
                # Metrics
                'participant_metrics': orig['participant_metrics'],
                'timestamp_metrics': orig['timestamp_metrics'],
                'reopen_metrics': orig.get('reopen_metrics'),
                
                # Gemini prediction (THE LABEL!)
                'predicted_label': pred['predicted_label'],
                'confidence': pred.get('confidence', 'Unknown'),
                'confidence_score': pred.get('confidence_score', 0.0),  # Default to 0.0 if missing
                'probabilities': pred.get('probabilities', {}),
                'reasoning': pred.get('reasoning', ''),
                'full_response': pred.get('full_response', ''),
                
                # Model metadata
                'model': pred.get('model'),
                'inference_time_seconds': pred.get('inference_time_seconds'),
                'timestamp': pred.get('timestamp'),
                
                # Ground truth (if exists)
                'final_classification': orig.get('final_classification'),
            }
            
            merged.append(merged_record)
            matched += 1
        else:
            unmatched_predictions.append({
                'project': project,
                'issue_number': issue_num,
                'title': pred.get('title', 'N/A')
            })
    
    # Save merged dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in merged:
            f.write(json.dumps(record) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"✅ MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Matched records: {matched}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")
    print(f"Match rate: {(matched/len(predictions)*100):.1f}%")
    print(f"Saved to: {output_path}")
    
    if unmatched_predictions:
        print(f"\n⚠️  Warning: {len(unmatched_predictions)} predictions had no matching original data")
        print(f"\nFirst 10 unmatched:")
        for i, item in enumerate(unmatched_predictions[:10], 1):
            print(f"  {i}. {item['project']} #{item['issue_number']}: {item['title'][:50]}")
    
    # Class distribution
    label_counts = defaultdict(int)
    confidence_by_label = defaultdict(list)
    
    for record in merged:
        label = record['predicted_label']
        label_counts[label] += 1
        # Only add confidence if it's not None/0.0
        conf_score = record.get('confidence_score')
        if conf_score and conf_score > 0:
            confidence_by_label[label].append(conf_score)
    
    print(f"\n{'='*60}")
    print(f"LABEL DISTRIBUTION")
    print(f"{'='*60}")
    for label in ['Intrinsic', 'Extrinsic', 'Not a Bug', 'Unknown']:
        if label in label_counts:
            count = label_counts[label]
            pct = (count / matched) * 100
            
            # Calculate average confidence if we have values
            if confidence_by_label[label]:
                avg_conf = sum(confidence_by_label[label]) / len(confidence_by_label[label])
                print(f"{label:15s}: {count:5d} ({pct:5.1f}%) - Avg confidence: {avg_conf:.3f}")
            else:
                print(f"{label:15s}: {count:5d} ({pct:5.1f}%) - No confidence scores")
    
    # Project distribution (top 10)
    project_counts = defaultdict(int)
    for record in merged:
        project_counts[record['project']] += 1
    
    print(f"\n{'='*60}")
    print(f"TOP 10 PROJECTS")
    print(f"{'='*60}")
    for project, count in sorted(project_counts.items(), key=lambda x: -x[1])[:10]:
        pct = (count / matched) * 100
        print(f"{project:40s}: {count:4d} ({pct:5.1f}%)")
    
    return merged

if __name__ == "__main__":
    # Run the merge
    merged = merge_datasets(
        original_path="issues_23659.jsonl",
        predictions_path="classified_23k_bugs.jsonl",
        output_path="merged_7k_training_data.jsonl"
    )
    
    # Show example record (just the key fields)
    print(f"\n{'='*60}")
    print(f"EXAMPLE MERGED RECORD (key fields)")
    print(f"{'='*60}")
    example = {
        'project': merged[0]['project'],
        'issue_number': merged[0]['issue_number'],
        'title': merged[0]['title'],
        'predicted_label': merged[0]['predicted_label'],
        'confidence_score': merged[0]['confidence_score'],
        'comments_count': merged[0]['comments_count'],
        'state': merged[0]['state'],
    }
    print(json.dumps(example, indent=2))