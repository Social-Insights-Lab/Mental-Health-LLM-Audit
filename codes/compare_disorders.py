#!/usr/bin/env python3
"""
Script to compare the expected disorders list with the actual disorders found in merged files
"""

import json
import os
from collections import Counter

def load_expected_disorders():
    """Load the expected disorders from the provided JSON file"""
    with open('/home/ad2688/Research/mental_disorder/filtered_combined_disorders.json', 'r', encoding='utf-8') as f:
        expected_disorders = json.load(f)
    
    # Normalize by stripping whitespace
    expected_disorders = [disorder.strip() for disorder in expected_disorders]
    return set(expected_disorders)

def get_actual_disorders():
    """Get disorders actually found in the merged files"""
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    
    # Get disorders from first merged file (they should all be the same)
    filepath = os.path.join(merged_dir, "merged_gemma-7b_generations.jsonl")
    
    actual_disorders = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    disorder = data.get('group', '').strip()
                    if disorder:
                        actual_disorders.add(disorder)
                except json.JSONDecodeError:
                    continue
    
    return actual_disorders

def main():
    """Compare expected vs actual disorders"""
    print("DISORDER COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load expected and actual disorders
    expected_disorders = load_expected_disorders()
    actual_disorders = get_actual_disorders()
    
    print(f"Expected disorders (from JSON file): {len(expected_disorders)}")
    print(f"Actual disorders (from merged files): {len(actual_disorders)}")
    
    # Find missing disorders
    missing_disorders = expected_disorders - actual_disorders
    
    # Find extra disorders (in files but not in expected list)
    extra_disorders = actual_disorders - expected_disorders
    
    print(f"\n📋 MISSING DISORDERS ({len(missing_disorders)}):")
    print("=" * 40)
    if missing_disorders:
        for disorder in sorted(missing_disorders):
            print(f"  - {disorder}")
    else:
        print("  ✅ No missing disorders!")
    
    print(f"\n📋 EXTRA DISORDERS ({len(extra_disorders)}):")
    print("=" * 40)
    if extra_disorders:
        for disorder in sorted(extra_disorders):
            print(f"  + {disorder}")
    else:
        print("  ✅ No extra disorders!")
    
    print(f"\n📊 SUMMARY:")
    print("=" * 40)
    print(f"Expected: {len(expected_disorders)} disorders")
    print(f"Found: {len(actual_disorders)} disorders") 
    print(f"Missing: {len(missing_disorders)} disorders")
    print(f"Extra: {len(extra_disorders)} disorders")
    
    if len(missing_disorders) == 0 and len(extra_disorders) == 0:
        print("🎉 Perfect match! All expected disorders are present.")
    elif len(missing_disorders) == 2 and len(extra_disorders) == 0:
        print("🔍 Found the 2 missing disorders that explain why we have 132 instead of 134 groups!")
    
    # Also check for potential duplicates in the expected list
    expected_list = json.load(open('/home/ad2688/Research/mental_disorder/filtered_combined_disorders.json', 'r'))
    expected_list_stripped = [disorder.strip() for disorder in expected_list]
    
    duplicates = []
    seen = set()
    for disorder in expected_list_stripped:
        if disorder in seen:
            duplicates.append(disorder)
        seen.add(disorder)
    
    if duplicates:
        print(f"\n⚠️  DUPLICATES IN EXPECTED LIST ({len(duplicates)}):")
        for duplicate in duplicates:
            print(f"  - {duplicate}")

if __name__ == "__main__":
    main()