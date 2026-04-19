#!/usr/bin/env python3
"""
Refined script to create exactly 100 samples per file, excluding UNKNOWN entries
"""

import json
import os
import random
import pandas as pd
from collections import Counter, defaultdict

def get_majority_vote(claude_label, openai_label, deepseek_label):
    """
    Get majority vote from three evaluator labels, excluding None values
    Returns: (majority_label, is_unanimous, vote_counts)
    """
    # Handle None values by treating them as abstentions
    valid_labels = [label for label in [claude_label, openai_label, deepseek_label] if label is not None]
    
    if not valid_labels:
        return None, False, {}  # Return None for UNKNOWN cases
    
    vote_counts = Counter(valid_labels)
    most_common = vote_counts.most_common()
    
    # Check if there's a clear majority
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        majority_label = most_common[0][0]
        is_unanimous = (most_common[0][1] == len(valid_labels))
    else:
        # In case of tie, prioritize in order: SUPPORT, REJECT, AMBIGUOUS
        priority_order = ["SUPPORT", "REJECT", "AMBIGUOUS"]
        for label in priority_order:
            if label in [item[0] for item in most_common if item[1] == most_common[0][1]]:
                majority_label = label
                break
        else:
            majority_label = most_common[0][0]
        is_unanimous = False
    
    return majority_label, is_unanimous, dict(vote_counts)

def process_file_refined(filepath):
    """
    Process a single merged file, excluding UNKNOWN entries
    Returns list of processed entries with valid majority votes
    """
    entries = []
    filename = os.path.basename(filepath)
    llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    
                    # Extract evaluation labels
                    evaluations = data.get('evaluations', {})
                    claude_eval = evaluations.get('Claude')
                    openai_eval = evaluations.get('OpenAI') 
                    deepseek_eval = evaluations.get('Deepseek')
                    
                    claude_label = claude_eval.get('label') if claude_eval else None
                    openai_label = openai_eval.get('label') if openai_eval else None
                    deepseek_label = deepseek_eval.get('label') if deepseek_eval else None
                    
                    # Get majority vote
                    majority_label, is_unanimous, vote_counts = get_majority_vote(
                        claude_label, openai_label, deepseek_label
                    )
                    
                    # Skip entries without valid majority vote (UNKNOWN)
                    if majority_label is None:
                        continue
                    
                    entry = {
                        'filename': llm_name,
                        'line_no': data.get('line_no', line_num),
                        'group': data.get('group', '').strip(),
                        'stereotype': data.get('stereotype', '').strip(),
                        'generation': data.get('generation', '').strip(),
                        'claude_label': claude_label,
                        'openai_label': openai_label,
                        'deepseek_label': deepseek_label,
                        'majority_label': majority_label,
                        'is_unanimous': is_unanimous,
                        'vote_counts': vote_counts
                    }
                    
                    entries.append(entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue
    
    return entries

def stratified_sample_exact(entries, target_size=100):
    """
    Sample exactly target_size entries with balanced representation across majority labels
    """
    # Group by majority label (should only be SUPPORT, REJECT, AMBIGUOUS now)
    label_groups = defaultdict(list)
    for entry in entries:
        label_groups[entry['majority_label']].append(entry)
    
    print(f"  Label distribution:")
    for label, group in label_groups.items():
        print(f"    {label}: {len(group)} entries")
    
    # Calculate target per group (aim for equal representation)
    available_labels = sorted(label_groups.keys())  # Sort for consistency
    target_per_label = target_size // len(available_labels)
    remainder = target_size % len(available_labels)
    
    sampled_entries = []
    
    # Sample from each label group
    for i, label in enumerate(available_labels):
        group = label_groups[label]
        # Add remainder to first few groups
        current_target = target_per_label + (1 if i < remainder else 0)
        
        if len(group) >= current_target:
            sampled = random.sample(group, current_target)
        else:
            # If not enough entries in this category, sample all and make up difference from other categories
            sampled = group
            print(f"    Warning: Only {len(group)} entries available for {label}, needed {current_target}")
        
        sampled_entries.extend(sampled)
        print(f"    Sampled {len(sampled)} entries for {label}")
    
    # If we're short of target due to insufficient entries in some categories,
    # sample additional entries from categories with surplus
    if len(sampled_entries) < target_size:
        shortage = target_size - len(sampled_entries)
        sampled_line_nos = {entry['line_no'] for entry in sampled_entries}
        
        # Get remaining entries not yet sampled
        remaining_entries = [entry for entry in entries if entry['line_no'] not in sampled_line_nos]
        
        if len(remaining_entries) >= shortage:
            additional_samples = random.sample(remaining_entries, shortage)
            sampled_entries.extend(additional_samples)
            print(f"    Added {shortage} additional samples to reach {target_size}")
    
    # Shuffle the final sample to mix labels
    random.shuffle(sampled_entries)
    return sampled_entries[:target_size]  # Ensure exactly target_size

def main():
    """Main function to create refined annotation sample"""
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    output_dir = "/home/ad2688/Research/mental_disorder/results_new/human_annotation"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("CREATING REFINED ANNOTATION SAMPLE (100 per file)")
    print("=" * 60)
    
    all_sampled_entries = []
    
    # Process each merged file
    for filename in sorted(os.listdir(merged_dir)):
        if filename.startswith('merged_') and filename.endswith('.jsonl'):
            llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
            filepath = os.path.join(merged_dir, filename)
            
            print(f"\nProcessing {llm_name}...")
            
            # Process file to get entries with valid majority voting (excluding UNKNOWN)
            entries = process_file_refined(filepath)
            print(f"  Total entries with valid labels: {len(entries)}")
            
            # Get majority label distribution
            label_dist = Counter([entry['majority_label'] for entry in entries])
            print(f"  Majority label distribution: {dict(label_dist)}")
            
            # Sample exactly 100 entries with balanced representation
            sampled_entries = stratified_sample_exact(entries, target_size=100)
            print(f"  Final sample size: {len(sampled_entries)}")
            
            # Verify the sample distribution
            sample_dist = Counter([entry['majority_label'] for entry in sampled_entries])
            print(f"  Sample distribution: {dict(sample_dist)}")
            
            all_sampled_entries.extend(sampled_entries)
    
    print(f"\n📊 OVERALL SAMPLING SUMMARY")
    print("=" * 40)
    print(f"Total sampled entries: {len(all_sampled_entries)}")
    
    # Show overall distribution
    overall_dist = Counter([entry['majority_label'] for entry in all_sampled_entries])
    print(f"Overall label distribution: {dict(overall_dist)}")
    
    # Show distribution by LLM
    llm_dist = Counter([entry['filename'] for entry in all_sampled_entries])
    print(f"Entries per LLM: {dict(llm_dist)}")
    
    # Create annotation files
    print(f"\n📝 CREATING FINAL ANNOTATION FILES")
    print("=" * 40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for annotation file
    annotation_data = []
    comparison_data = []
    
    for entry in all_sampled_entries:
        # For human annotation file
        annotation_row = {
            'Filename': entry['filename'],
            'Line_No': entry['line_no'],
            'Group': entry['group'],
            'Stereotype': entry['stereotype'],
            'Generation': entry['generation'][:1000] + "..." if len(entry['generation']) > 1000 else entry['generation'],
            'Human_Annot': ''  # Empty for human to fill
        }
        annotation_data.append(annotation_row)
        
        # For comparison file
        comparison_row = {
            'Filename': entry['filename'],
            'Line_No': entry['line_no'],
            'Group': entry['group'],
            'Stereotype': entry['stereotype'],
            'OpenAI_Label': entry['openai_label'],
            'Claude_Label': entry['claude_label'], 
            'Deepseek_Label': entry['deepseek_label'],
            'Majority_Label': entry['majority_label'],
            'Is_Unanimous': entry['is_unanimous'],
            'Vote_Counts': str(entry['vote_counts'])
        }
        comparison_data.append(comparison_row)
    
    # Create DataFrames
    annotation_df = pd.DataFrame(annotation_data)
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to Excel files
    annotation_file = os.path.join(output_dir, 'human_annotation_sample_refined.xlsx')
    comparison_file = os.path.join(output_dir, 'llm_labels_comparison_refined.xlsx')
    
    with pd.ExcelWriter(annotation_file, engine='openpyxl') as writer:
        annotation_df.to_excel(writer, sheet_name='Annotation', index=False)
    
    with pd.ExcelWriter(comparison_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
    
    print(f"📄 Created annotation file: {annotation_file}")
    print(f"📄 Created comparison file: {comparison_file}")
    
    print(f"\n✅ COMPLETED!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Each LLM now has exactly 100 sampled entries for human annotation")

if __name__ == "__main__":
    main()