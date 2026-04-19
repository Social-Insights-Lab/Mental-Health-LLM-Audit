#!/usr/bin/env python3
"""
Script to verify that all files have exactly 45 entries per group
"""

import json
import os
from collections import Counter

def verify_file_structure(filepath):
    """Verify file has exactly 45 entries per group"""
    group_counts = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    group = data.get('group', '').strip()
                    group_counts[group] += 1
                except json.JSONDecodeError:
                    continue
    
    # Check if all groups have exactly 45 entries
    correct_structure = True
    issues = []
    
    for group, count in group_counts.items():
        if count != 45:
            correct_structure = False
            issues.append(f"{group}: {count}")
    
    return correct_structure, len(group_counts), sum(group_counts.values()), issues

def main():
    """Verify all merged files have correct structure"""
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    
    print("VERIFYING FILTERED FILE STRUCTURE")
    print("=" * 60)
    print("Expected: 132 groups × 45 entries = 5,940 total entries per file")
    print()
    
    all_correct = True
    
    for filename in sorted(os.listdir(merged_dir)):
        if filename.startswith('merged_') and filename.endswith('.jsonl'):
            llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
            filepath = os.path.join(merged_dir, filename)
            
            correct, groups, total, issues = verify_file_structure(filepath)
            
            status = "✅ CORRECT" if correct else "❌ ISSUES"
            print(f"{llm_name}: {status}")
            print(f"  Groups: {groups}, Total entries: {total}")
            
            if issues:
                all_correct = False
                print(f"  Issues found:")
                for issue in issues:
                    print(f"    {issue}")
    
    print()
    if all_correct:
        print("🎉 All files have the correct structure!")
        print("📊 Each file contains exactly 45 entries per group across 132 groups")
    else:
        print("⚠️  Some files still have structural issues")

if __name__ == "__main__":
    main()