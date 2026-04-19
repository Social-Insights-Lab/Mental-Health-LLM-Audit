#!/usr/bin/env python3
"""
Script to merge generation files with their evaluations from Claude, OpenAI, and Deepseek.
Each merged file will contain generation data with all available evaluations.
"""

import json
import os
from typing import Dict, List, Optional

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into a list of dictionaries"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """Save list of dictionaries to JSONL file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_llm_data(base_dir: str, llm_name: str) -> List[Dict]:
    """
    Merge generation file with evaluation files for a specific LLM
    
    Args:
        base_dir: Base directory containing the files
        llm_name: Name of the LLM (e.g., 'gemma-7b')
    
    Returns:
        List of merged dictionaries
    """
    # Load generation file
    gen_file = os.path.join(base_dir, f"{llm_name}_generations.jsonl")
    generations = load_jsonl(gen_file)
    
    if not generations:
        print(f"No generation data found for {llm_name}")
        return []
    
    print(f"Loaded {len(generations)} generations for {llm_name}")
    
    # Load evaluation files
    evaluators = ['Claude', 'OpenAI', 'Deepseek']
    evaluations = {}
    
    for evaluator in evaluators:
        eval_file = os.path.join(base_dir, evaluator, f"{evaluator}_evaluation_{llm_name}_generations.jsonl")
        eval_data = load_jsonl(eval_file)
        
        # Create a lookup dictionary by (line_no, group) for fast access
        evaluations[evaluator] = {}
        for eval_item in eval_data:
            key = (eval_item.get('line_no'), eval_item.get('group', '').strip())
            evaluations[evaluator][key] = eval_item['evaluation']
        
        print(f"Loaded {len(eval_data)} evaluations from {evaluator} for {llm_name}")
    
    # Merge data
    merged_data = []
    
    for i, gen_item in enumerate(generations, 1):  # Line numbers start from 1
        # Create merged entry starting with generation data
        merged_entry = gen_item.copy()
        merged_entry['line_no'] = i  # Add line number for reference
        
        # Add evaluations from each evaluator
        merged_entry['evaluations'] = {}
        
        key = (i, gen_item.get('group', '').strip())
        
        for evaluator in evaluators:
            if key in evaluations[evaluator]:
                merged_entry['evaluations'][evaluator] = evaluations[evaluator][key]
            else:
                merged_entry['evaluations'][evaluator] = None
        
        merged_data.append(merged_entry)
    
    return merged_data

def main():
    """Main function to process all LLM files"""
    base_dir = "/home/ad2688/Research/mental_disorder/results_new"
    
    # Get list of all generation files
    gen_files = [f for f in os.listdir(base_dir) if f.endswith('_generations.jsonl')]
    
    # Extract LLM names
    llm_names = []
    for gen_file in gen_files:
        llm_name = gen_file.replace('_generations.jsonl', '')
        llm_names.append(llm_name)
    
    print(f"Found {len(llm_names)} LLMs to process:")
    for llm_name in sorted(llm_names):
        print(f"  - {llm_name}")
    
    print("\nStarting merge process...")
    
    # Process each LLM
    for llm_name in sorted(llm_names):
        print(f"\nProcessing {llm_name}...")
        
        merged_data = merge_llm_data(base_dir, llm_name)
        
        if merged_data:
            # Save merged file
            output_file = os.path.join(base_dir, "merged", f"merged_{llm_name}_generations.jsonl")
            save_jsonl(merged_data, output_file)
            
            # Count evaluations
            total_with_evaluations = 0
            evaluator_counts = {'Claude': 0, 'OpenAI': 0, 'Deepseek': 0}
            
            for entry in merged_data:
                has_any_eval = False
                for evaluator in ['Claude', 'OpenAI', 'Deepseek']:
                    if entry['evaluations'][evaluator] is not None:
                        evaluator_counts[evaluator] += 1
                        has_any_eval = True
                if has_any_eval:
                    total_with_evaluations += 1
            
            print(f"  Saved {len(merged_data)} merged entries to {output_file}")
            print(f"  Entries with evaluations: {total_with_evaluations}")
            print(f"  Evaluation counts: Claude={evaluator_counts['Claude']}, OpenAI={evaluator_counts['OpenAI']}, Deepseek={evaluator_counts['Deepseek']}")
        else:
            print(f"  No data to save for {llm_name}")
    
    print("\nMerge process completed!")

if __name__ == "__main__":
    main()