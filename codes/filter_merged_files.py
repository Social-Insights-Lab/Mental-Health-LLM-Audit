#!/usr/bin/env python3
"""
Script to analyze and filter merged files to ensure exactly 45 entries per group
(3 instruction prompts × 15 stereotypes = 45 entries per mental health group)
"""

import json
import os
from collections import defaultdict, Counter

def analyze_file_groups(filepath):
    """Analyze group distribution in a file"""
    group_counts = Counter()
    stereotype_counts = defaultdict(Counter)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    group = data.get('group', '').strip()
                    stereotype = data.get('stereotype', '').strip()
                    
                    group_counts[group] += 1
                    stereotype_counts[group][stereotype] += 1
                    
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_no} in {filepath}")
                    continue
    
    return group_counts, stereotype_counts

def filter_file_to_exact_counts(input_filepath, output_filepath, target_per_group=45):
    """Filter file to keep exactly target_per_group entries per group"""
    
    # First pass: collect all entries by group
    entries_by_group = defaultdict(list)
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    group = data.get('group', '').strip()
                    entries_by_group[group].append(data)
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_no} in {input_filepath}")
                    continue
    
    # Filter to keep exactly target_per_group entries per group
    filtered_entries = []
    stats = {'groups_found': 0, 'groups_filtered': 0, 'total_entries': 0}
    
    for group, entries in entries_by_group.items():
        stats['groups_found'] += 1
        
        if len(entries) > target_per_group:
            # Keep first target_per_group entries
            filtered_entries.extend(entries[:target_per_group])
            stats['groups_filtered'] += 1
            print(f"  {group}: {len(entries)} → {target_per_group} entries")
        else:
            # Keep all entries if less than or equal to target
            filtered_entries.extend(entries)
            if len(entries) < target_per_group:
                print(f"  {group}: {len(entries)} entries (less than {target_per_group})")
    
    stats['total_entries'] = len(filtered_entries)
    
    # Save filtered entries
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return stats

def main():
    """Main function to analyze and filter all merged files"""
    merged_dir = "/home/ad2688/Research/mental_disorder/results_new/merged"
    
    print("ANALYZING CURRENT GROUP DISTRIBUTIONS")
    print("=" * 60)
    
    files_to_process = []
    for filename in sorted(os.listdir(merged_dir)):
        if filename.startswith('merged_') and filename.endswith('.jsonl'):
            llm_name = filename.replace('merged_', '').replace('_generations.jsonl', '')
            filepath = os.path.join(merged_dir, filename)
            
            print(f"\n{llm_name}:")
            group_counts, stereotype_counts = analyze_file_groups(filepath)
            
            total_entries = sum(group_counts.values())
            unique_groups = len(group_counts)
            
            print(f"  Total entries: {total_entries}")
            print(f"  Unique groups: {unique_groups}")
            
            if total_entries != 6030 or unique_groups != 134:
                files_to_process.append((filepath, llm_name, total_entries, unique_groups))
                print(f"  ⚠️  NEEDS FILTERING (expected 6030 entries, 134 groups)")
                
                # Show groups with unusual counts
                for group, count in group_counts.most_common():
                    if count != 45:
                        print(f"    {group}: {count} entries")
            else:
                print(f"  ✅ CORRECT (6030 entries, 134 groups)")
    
    if not files_to_process:
        print(f"\n🎉 All files already have the correct structure!")
        return
    
    print(f"\n\nFILTERING FILES")
    print("=" * 60)
    
    for filepath, llm_name, total_entries, unique_groups in files_to_process:
        print(f"\nProcessing {llm_name}...")
        
        # Create backup
        backup_path = filepath + '.backup'
        os.rename(filepath, backup_path)
        
        # Filter and save
        stats = filter_file_to_exact_counts(backup_path, filepath, target_per_group=45)
        
        print(f"  Groups found: {stats['groups_found']}")
        print(f"  Groups filtered: {stats['groups_filtered']}")
        print(f"  Final entries: {stats['total_entries']}")
        
        if stats['total_entries'] == 6030 and stats['groups_found'] == 134:
            print(f"  ✅ Successfully filtered to target size")
        else:
            print(f"  ⚠️  Final size: {stats['total_entries']} (expected 6030)")
    
    print(f"\n✅ Filtering completed!")
    print(f"📝 Original files backed up with .backup extension")

if __name__ == "__main__":
    main()