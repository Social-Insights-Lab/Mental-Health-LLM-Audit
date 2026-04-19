# Human Annotation Sample - Final Summary

## Overview
Successfully created human annotation sample with **exactly 100 entries per LLM** for comprehensive evaluation comparison.

## Files Created
📁 **Location**: `/home/ad2688/Research/mental_disorder/results_new/human_annotation/`

### 1. `human_annotation_sample_refined.xlsx`
**For Human Annotation**
- **Total entries**: 1,200 (100 per LLM × 12 LLMs)
- **Columns**: 
  - `Filename`: LLM name
  - `Line_No`: Original line number in merged file
  - `Group`: Mental health condition
  - `Stereotype`: Stereotype being evaluated
  - `Generation`: LLM's text generation (truncated to 1000 chars for readability)
  - `Human_Annot`: **EMPTY** - for you to fill with your annotations

### 2. `llm_labels_comparison_refined.xlsx`
**For Comparison with LLM Evaluations**
- **Total entries**: 1,200 (same sample as above)
- **Columns**:
  - `Filename`: LLM name  
  - `Line_No`: Original line number
  - `Group`: Mental health condition
  - `Stereotype`: Stereotype being evaluated
  - `OpenAI_Label`: OpenAI's evaluation label
  - `Claude_Label`: Claude's evaluation label
  - `Deepseek_Label`: Deepseek's evaluation label
  - `Majority_Label`: Majority vote result
  - `Is_Unanimous`: Whether all 3 evaluators agreed
  - `Vote_Counts`: Count of each vote type

## Sampling Strategy

### 1. Majority Voting
- **Method**: Used majority vote from Claude, OpenAI, and Deepseek evaluations
- **Tie-breaking**: Priority order: SUPPORT → REJECT → AMBIGUOUS
- **Exclusions**: Removed entries where all 3 evaluators returned null (UNKNOWN)

### 2. Balanced Sampling  
- **Per LLM**: Exactly 100 entries
- **Distribution**: Approximately equal representation across:
  - **SUPPORT**: ~33 entries per LLM
  - **REJECT**: ~33 entries per LLM  
  - **AMBIGUOUS**: ~34 entries per LLM
- **Random seed**: 42 (for reproducibility)

## Final Distribution

### Overall Label Balance
- **AMBIGUOUS**: 408 entries (34%)
- **REJECT**: 396 entries (33%) 
- **SUPPORT**: 396 entries (33%)

### Per-LLM Coverage
All 12 LLMs have exactly **100 entries** each:
- gemma-7b, gemma2-9b, gemma3-4b-uncensored
- llama2-7B, llama3, llama3-70B, llama3.1-8B-it-uncensored  
- mistral-uncensored, mistral-v0.1, mistral-v0.2, mistral-v0.3
- mixtral-8x22b

## Usage Instructions

### For Human Annotation
1. Open `human_annotation_sample_refined.xlsx`
2. Fill the `Human_Annot` column with your labels: **SUPPORT**, **REJECT**, or **AMBIGUOUS**
3. Use the same criteria as the LLM evaluators for consistency

### For Analysis
1. After completing human annotation, use `llm_labels_comparison_refined.xlsx`
2. Compare your `Human_Annot` labels with:
   - Individual LLM labels (OpenAI, Claude, Deepseek)
   - `Majority_Label` for overall LLM consensus
   - `Is_Unanimous` to see agreement levels

## Quality Assurance
✅ **Perfect balance**: Each LLM contributes exactly 100 entries  
✅ **Representative sampling**: Balanced across SUPPORT/REJECT/AMBIGUOUS  
✅ **No missing data**: Excluded entries with null evaluations  
✅ **Traceable**: Line numbers preserved for reference back to original files  
✅ **Reproducible**: Fixed random seed ensures consistent sampling  

## Next Steps
1. **Human Annotation**: Fill in the `Human_Annot` column in the annotation file
2. **Analysis**: Compare human annotations with LLM majority voting
3. **Evaluation**: Calculate agreement metrics (Cohen's kappa, etc.)
4. **Insights**: Identify patterns in human vs. LLM disagreements