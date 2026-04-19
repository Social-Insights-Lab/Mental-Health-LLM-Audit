#!/usr/bin/env python3
"""
Script to analyze null evaluations in merged files
"""

import json
import os
from collections import defaultdict

def analyze_null_evaluations(merged_dir):
    """Analyze how many null evaluations exist in merged files"""
    
    results = {}
    
    for filename in os.listdir(merged_dir):
        if filename.startswith('merged_') and filename.endswith('.jsonl'):
            llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
            filepath = os.path.join(merged_dir, filename)
            
            print(f"\nAnalyzing {llm_name}...")
            
            total_entries = 0
            null_counts = {'Claude': 0, 'OpenAI': 0, 'Deepseek': 0}
            entries_with_any_null = 0
            entries_with_all_null = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if line.strip():
                        total_entries += 1
                        data = json.loads(line)
                        
                        evaluations = data.get('evaluations', {})
                        has_null = False
                        all_null = True
                        
                        for evaluator in ['Claude', 'OpenAI', 'Deepseek']:
                            if evaluations.get(evaluator) is None:
                                null_counts[evaluator] += 1
                                has_null = True
                            else:
                                all_null = False
                        
                        if has_null:
                            entries_with_any_null += 1
                        if all_null:
                            entries_with_all_null += 1
                            print(f"  Line {line_no}: ALL NULL - Group: {data.get('group')}, Line_no: {data.get('line_no')}")
            
            results[llm_name] = {
                'total_entries': total_entries,
                'null_counts': null_counts,
                'entries_with_any_null': entries_with_any_null,
                'entries_with_all_null': entries_with_all_null
            }
            
            print(f"  Total entries: {total_entries}")
            print(f"  Null evaluations - Claude: {null_counts['Claude']}, OpenAI: {null_counts['OpenAI']}, Deepseek: {null_counts['Deepseek']}")
            print(f"  Entries with any null evaluation: {entries_with_any_null}")
            print(f"  Entries with all null evaluations: {entries_with_all_null}")
    
    return results

def main():
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    results = analyze_null_evaluations(merged_dir)
    
    print("\n" + "="*60)
    print("SUMMARY OF NULL EVALUATIONS")
    print("="*60)
    
    for llm_name, data in results.items():
        print(f"\n{llm_name}:")
        print(f"  Total: {data['total_entries']}")
        print(f"  Entries with nulls: {data['entries_with_any_null']} ({data['entries_with_any_null']/data['total_entries']*100:.2f}%)")
        print(f"  All nulls: {data['entries_with_all_null']}")

if __name__ == "__main__":
    main()