#!/usr/bin/env python3
"""
Script to identify which mental health groups are missing from the files
"""

import json
import os
from collections import Counter

def get_groups_from_file(filepath):
    """Get unique groups from a file"""
    groups = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    group = data.get('group', '').strip()
                    if group:
                        groups.add(group)
                except json.JSONDecodeError:
                    continue
    
    return groups

def main():
    """Find missing groups across all files"""
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    
    # Get groups from all files
    all_file_groups = {}
    
    for filename in sorted(os.listdir(merged_dir)):
        if filename.startswith('merged_') and filename.endswith('.jsonl'):
            llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
            filepath = os.path.join(merged_dir, filename)
            
            groups = get_groups_from_file(filepath)
            all_file_groups[llm_name] = groups
            print(f"{llm_name}: {len(groups)} unique groups")
    
    # Find the union of all groups across all files
    all_groups_union = set()
    for groups in all_file_groups.values():
        all_groups_union.update(groups)
    
    print(f"\nTotal unique groups across all files: {len(all_groups_union)}")
    
    # Show which groups are missing from each file
    for llm_name, groups in all_file_groups.items():
        missing = all_groups_union - groups
        if missing:
            print(f"\n{llm_name} missing groups:")
            for group in sorted(missing):
                print(f"  - {group}")
        else:
            print(f"\n{llm_name}: No missing groups")
    
    # Show all unique groups found
    print(f"\nAll unique groups found ({len(all_groups_union)}):")
    for group in sorted(all_groups_union):
        print(f"  - {group}")

if __name__ == "__main__":
    main()