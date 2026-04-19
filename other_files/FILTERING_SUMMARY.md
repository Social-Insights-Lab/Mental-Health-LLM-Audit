# Filtering Summary Report

## What Was Done
Successfully filtered all merged files to ensure consistent structure with exactly 45 entries per mental health group.

## Process Steps
1. **Analysis**: Identified files with inconsistent entry counts and group distributions
2. **Backup**: Created `.backup` copies of original merged files  
3. **Filtering**: Kept only the first 45 entries for each mental health group
4. **Verification**: Confirmed all files now have identical structure

## Before Filtering
- Variable entry counts: 5,940 to 7,076 entries per file
- Inconsistent groups per file: All had 132 groups but varying entries per group
- Some groups had 45-225 entries (should be exactly 45)

## After Filtering  
- Consistent entry counts: Exactly 5,940 entries per file
- Consistent structure: 132 groups × 45 entries per group
- Perfect alignment: 3 instruction prompts × 15 stereotypes = 45 per group

## Files Processed
All 12 LLM merged files were successfully filtered:
- `merged_gemma-7b_generations.jsonl`
- `merged_gemma2-9b_generations.jsonl`
- `merged_gemma3-4b-uncensored_generations.jsonl`
- `merged_llama2-7B_generations.jsonl`
- `merged_llama3_generations.jsonl`
- `merged_llama3-70B_generations.jsonl`
- `merged_llama3.1-8B-it-uncensored_generations.jsonl`
- `merged_mistral-uncensored_generations.jsonl`
- `merged_mistral-v0.1_generations.jsonl`
- `merged_mistral-v0.2_generations.jsonl`
- `merged_mistral-v0.3_generations.jsonl`
- `merged_mixtral-8x22b_generations.jsonl`

## Note on 132 vs 134 Groups
- **Expected**: 134 mental health groups
- **Found**: 132 mental health groups in all generation files
- **Conclusion**: Original data was generated with only 132 groups, not 134
- **Result**: 132 × 45 = 5,940 entries per file (not 6,030 as originally expected)

## Data Preservation
- Original merged files backed up as `*.backup` files
- Original generation and evaluation files remain untouched
- Only the working merged files have been filtered for consistency
- All evaluations preserved during filtering process

## Verification
✅ All files verified to have exactly 45 entries per group  
✅ All files contain exactly 132 unique mental health groups  
✅ Evaluation coverage remains excellent (>99.9%)  
✅ Data structure is now perfectly consistent across all 12 LLMs