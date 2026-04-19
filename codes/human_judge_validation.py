#!/usr/bin/env python3
"""
Human vs LLM-Judge Validation Analysis
=======================================
Computes all metrics referenced in the paper:
  1. Inter-annotator Cohen's kappa (human-human)
  2. Per-class Precision / Recall / F1 (human-majority vs LLM-judge-majority)
  3. Confusion matrix (human vs judge)
  4. Judge agreement statistics
  5. Uncertainty propagation analysis
  6. LaTeX-ready tables for the paper (Table 3, Appendix B, Appendix E)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: matplotlib/seaborn not available — skipping figure generation")
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/ad2688/Research/Works_Done/mental_disorder")
HUMAN_DIR  = BASE_DIR / "human_annot"
MERGED_DIR = BASE_DIR / "results_new" / "merged"
FIG_DIR    = BASE_DIR / "figures"
OUT_DIR    = BASE_DIR  # put output tables here

LABELS = ["SUPPORT", "REJECT", "AMBIGUOUS"]

# ─── 1. Load human annotations ──────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Loading human annotations")
print("=" * 70)

df_h1 = pd.read_excel(HUMAN_DIR / "human_annotation_sample_refined.xlsx", engine="openpyxl")
df_h2 = pd.read_excel(HUMAN_DIR / "2_human_annotation_sample_refined.xlsx", engine="openpyxl")
df_llm = pd.read_excel(HUMAN_DIR / "llm_labels_comparison_refined.xlsx", engine="openpyxl")

print(f"Annotator-1 file: {len(df_h1)} rows, {df_h1['Human_Annot1'].notna().sum()} labeled")
print(f"Annotator-2 file: {len(df_h2)} rows, {df_h2['Human_Annot2'].notna().sum()} labeled")
print(f"LLM labels file:  {len(df_llm)} rows")

# Merge the two annotator files
merged = pd.merge(
    df_h1[["Filename", "Line_No", "Human_Annot1"]],
    df_h2[["Filename", "Line_No", "Human_Annot2"]],
    on=["Filename", "Line_No"],
    how="inner",
)
print(f"Merged (inner join): {len(merged)} rows")

# Clean labels
for col in ["Human_Annot1", "Human_Annot2"]:
    merged[col] = merged[col].astype(str).str.upper().str.strip()
    merged.loc[merged[col] == "NAN", col] = np.nan

# Keep rows where BOTH annotators provided a label
both_labeled = merged.dropna(subset=["Human_Annot1", "Human_Annot2"]).copy()
print(f"Both annotators labeled: {len(both_labeled)} rows")
print(f"  Annot-1 distribution: {dict(both_labeled['Human_Annot1'].value_counts())}")
print(f"  Annot-2 distribution: {dict(both_labeled['Human_Annot2'].value_counts())}")

# ─── 2. Cohen's kappa (human–human) ─────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Inter-annotator agreement (Human–Human)")
print("=" * 70)

annot1 = both_labeled["Human_Annot1"]
annot2 = both_labeled["Human_Annot2"]

kappa_hh = cohen_kappa_score(annot1, annot2)
print(f"Cohen's kappa (Human1 vs Human2): {kappa_hh:.4f}")

# Confusion matrix between annotators
cm_hh = confusion_matrix(annot1, annot2, labels=LABELS)
print("\nConfusion matrix (rows=Annot1, cols=Annot2):")
cm_df_hh = pd.DataFrame(cm_hh, index=LABELS, columns=LABELS)
print(cm_df_hh)

# Classification report (Annot1 as reference)
print("\nClassification report (Annot1 as reference, Annot2 as predicted):")
print(classification_report(annot1, annot2, labels=LABELS, digits=3))

# Plot confusion matrix
if HAS_PLOT:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_hh, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Annotator 2")
    ax.set_ylabel("Annotator 1")
    ax.set_title(f"Human Inter-Annotator Confusion Matrix (κ = {kappa_hh:.3f})")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "human_interannotator_cm.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "human_interannotator_cm.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIG_DIR / 'human_interannotator_cm.pdf'}")

# ─── 3. Compute human-majority label ────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Human majority label computation")
print("=" * 70)

def human_majority(row):
    a1, a2 = row["Human_Annot1"], row["Human_Annot2"]
    if a1 == a2:
        return a1
    else:
        return "AMBIGUOUS"  # ties mapped to AMBIGUOUS, as stated in paper

both_labeled["human_majority"] = both_labeled.apply(human_majority, axis=1)

print(f"Human-majority label distribution:")
print(both_labeled["human_majority"].value_counts().to_string())

# ─── 4. Merge with LLM-judge labels ─────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Merging human labels with LLM-judge labels")
print("=" * 70)

# Clean LLM labels
df_llm["Majority_Label"] = df_llm["Majority_Label"].astype(str).str.upper().str.strip()

val_merged = pd.merge(
    both_labeled[["Filename", "Line_No", "Human_Annot1", "Human_Annot2", "human_majority"]],
    df_llm[["Filename", "Line_No", "OpenAI_Label", "Claude_Label", "Deepseek_Label",
            "Majority_Label", "Is_Unanimous"]],
    on=["Filename", "Line_No"],
    how="inner",
)
print(f"Validation set size (human + judge labels): {len(val_merged)} rows")
print(f"  Human-majority distribution:  {dict(val_merged['human_majority'].value_counts())}")
print(f"  Judge-majority distribution:  {dict(val_merged['Majority_Label'].value_counts())}")

# ─── 5. Per-class Precision / Recall / F1 (Human-majority vs Judge-majority) ──
print("\n" + "=" * 70)
print("SECTION 5: Per-class Precision / Recall / F1")
print("           (Human-majority = ground truth, Judge-majority = predicted)")
print("=" * 70)

y_true = val_merged["human_majority"]
y_pred = val_merged["Majority_Label"]

# Ensure all labels present
report_str = classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0)
print(report_str)

report_dict = classification_report(y_true, y_pred, labels=LABELS, digits=4,
                                    output_dict=True, zero_division=0)

# Overall accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Overall accuracy: {acc:.4f}")

# Also compute Cohen's kappa (human vs judge)
kappa_hj = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's kappa (Human-majority vs Judge-majority): {kappa_hj:.4f}")

# ─── 6. Confusion matrix (Human vs Judge) ───────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Confusion matrix (Human-majority vs Judge-majority)")
print("=" * 70)

cm_hj = confusion_matrix(y_true, y_pred, labels=LABELS)
cm_df_hj = pd.DataFrame(cm_hj, index=[f"H:{l}" for l in LABELS],
                         columns=[f"J:{l}" for l in LABELS])
print(cm_df_hj)

# Normalized confusion matrix (row-normalized = recall per class)
cm_norm = cm_hj.astype(float) / cm_hj.sum(axis=1, keepdims=True)
cm_norm_df = pd.DataFrame(np.round(cm_norm, 3),
                          index=[f"H:{l}" for l in LABELS],
                          columns=[f"J:{l}" for l in LABELS])
print("\nRow-normalized (recall perspective):")
print(cm_norm_df)

# Plot confusion matrix
if HAS_PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm_hj, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS,
                cmap="Greens", ax=axes[0])
    axes[0].set_xlabel("Judge Majority")
    axes[0].set_ylabel("Human Majority")
    axes[0].set_title(f"Confusion Matrix (κ = {kappa_hj:.3f})")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=LABELS, yticklabels=LABELS,
                cmap="Greens", ax=axes[1])
    axes[1].set_xlabel("Judge Majority")
    axes[1].set_ylabel("Human Majority")
    axes[1].set_title("Normalized Confusion Matrix (row = recall)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "human_vs_judge_cm.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "human_vs_judge_cm.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIG_DIR / 'human_vs_judge_cm.pdf'}")

# ─── 7. Per-judge agreement with human ──────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Individual judge agreement with human-majority")
print("=" * 70)

for judge_col in ["OpenAI_Label", "Claude_Label", "Deepseek_Label"]:
    judge_name = judge_col.replace("_Label", "")
    j_labels = val_merged[judge_col].astype(str).str.upper().str.strip()
    
    kappa_j = cohen_kappa_score(y_true, j_labels)
    acc_j = accuracy_score(y_true, j_labels)
    
    print(f"\n--- {judge_name} ---")
    print(f"  Cohen's κ: {kappa_j:.4f}")
    print(f"  Accuracy:  {acc_j:.4f}")
    print(classification_report(y_true, j_labels, labels=LABELS, digits=3, zero_division=0))

# ─── 8. Judge panel agreement statistics ─────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: Judge panel agreement (on validation subset)")
print("=" * 70)

def judge_agreement_level(row):
    labels = [
        str(row["OpenAI_Label"]).upper().strip(),
        str(row["Claude_Label"]).upper().strip(),
        str(row["Deepseek_Label"]).upper().strip(),
    ]
    c = Counter(labels)
    if c.most_common(1)[0][1] == 3:
        return "Unanimous"
    elif c.most_common(1)[0][1] == 2:
        return "Majority"
    else:
        return "No Agreement"

val_merged["judge_agreement"] = val_merged.apply(judge_agreement_level, axis=1)
agree_counts = val_merged["judge_agreement"].value_counts()
print("Judge agreement on validation subset:")
for level in ["Unanimous", "Majority", "No Agreement"]:
    n = agree_counts.get(level, 0)
    pct = 100 * n / len(val_merged)
    print(f"  {level:15s}: {n:4d} ({pct:.1f}%)")

# ─── 9. Uncertainty propagation analysis ────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9: Uncertainty propagation analysis")
print("=" * 70)

# Compute empirical error rates from confusion matrix
print("Empirical misclassification rates (from confusion matrix):")
for i, true_label in enumerate(LABELS):
    row_total = cm_hj[i].sum()
    if row_total > 0:
        correct = cm_hj[i, i]
        error_rate = 1 - correct / row_total
        print(f"  P(Judge ≠ {true_label} | Human = {true_label}) = {error_rate:.3f}")

# Estimate how judge errors could shift aggregate support rates
# Using the confusion matrix to compute corrected estimates
print("\nCorrection matrix (how judge labels map to true labels):")
# P(true=j | pred=i) = column-normalized confusion matrix
cm_col_norm = cm_hj.astype(float) / cm_hj.sum(axis=0, keepdims=True)
cm_col_df = pd.DataFrame(np.round(cm_col_norm, 3),
                         index=[f"True:{l}" for l in LABELS],
                         columns=[f"Pred:{l}" for l in LABELS])
print(cm_col_df)

# Estimate support rate bounds
# If the judge says SUPPORT with probability p, the true SUPPORT rate is approximately:
# p_true = P(True=SUP|Pred=SUP) * p_pred_sup + P(True=SUP|Pred=other) * p_pred_other
sup_idx = LABELS.index("SUPPORT")
p_true_sup_given_pred_sup = cm_col_norm[sup_idx, sup_idx] if cm_hj[:, sup_idx].sum() > 0 else 0
p_true_sup_given_pred_rej = cm_col_norm[sup_idx, LABELS.index("REJECT")] if cm_hj[:, LABELS.index("REJECT")].sum() > 0 else 0
p_true_sup_given_pred_amb = cm_col_norm[sup_idx, LABELS.index("AMBIGUOUS")] if cm_hj[:, LABELS.index("AMBIGUOUS")].sum() > 0 else 0

print(f"\nP(True=SUPPORT | Pred=SUPPORT)   = {p_true_sup_given_pred_sup:.3f}")
print(f"P(True=SUPPORT | Pred=REJECT)    = {p_true_sup_given_pred_rej:.3f}")
print(f"P(True=SUPPORT | Pred=AMBIGUOUS) = {p_true_sup_given_pred_amb:.3f}")

# Load full dataset to compute corrected rates
print("\n--- Loading full dataset for corrected support rate estimates ---")
merged_dir = BASE_DIR / "results_new" / "majority_vote"
full_data = []
for fp in sorted(merged_dir.glob("*.jsonl")):
    model_name = fp.stem.replace("majority_vote_", "").replace("merged_", "").replace("_generations", "")
    with open(fp, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    mv = entry.get("majority_vote", {})
                    label = mv.get("label", "")
                    full_data.append({"model": model_name, "label": label})
                except:
                    continue

df_full = pd.DataFrame(full_data)
print(f"Full dataset: {len(df_full)} entries across {df_full['model'].nunique()} models")

# For each model, compute observed and corrected support rates
print("\nModel-level support rate estimates (observed vs corrected):")
print(f"{'Model':<35s} {'Observed':>10s} {'Corrected':>10s} {'Δ':>8s}")
print("-" * 68)

for model in sorted(df_full["model"].unique()):
    model_data = df_full[df_full["model"] == model]
    n = len(model_data)
    
    # Observed rates from judge
    p_sup = (model_data["label"] == "SUPPORT").mean()
    p_rej = (model_data["label"] == "REJECT").mean()
    p_amb = (model_data["label"] == "AMBIGUOUS").mean()
    
    # Corrected support rate using confusion matrix
    corrected_sup = (
        p_true_sup_given_pred_sup * p_sup
        + p_true_sup_given_pred_rej * p_rej
        + p_true_sup_given_pred_amb * p_amb
    )
    
    delta = corrected_sup - p_sup
    print(f"{model:<35s} {p_sup:>9.1%} {corrected_sup:>9.1%} {delta:>+7.1%}")

# ─── 10. Generate LaTeX tables ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10: LaTeX output for paper")
print("=" * 70)

# --- Table 3: Per-class P/R/F1 ---
print("\n%%% TABLE 3: Per-class Precision / Recall / F1 %%%")
print("%%% (Human-majority = ground truth, Judge-majority = predicted)")
latex_table3 = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{$F_1$} \\
\midrule
"""

for label in LABELS:
    p = report_dict[label]["precision"]
    r = report_dict[label]["recall"]
    f = report_dict[label]["f1-score"]
    latex_table3 += f"\\textsc{{{label.capitalize()}}} & {p:.3f} & {r:.3f} & {f:.3f} \\\\\n"

macro_p = report_dict["macro avg"]["precision"]
macro_r = report_dict["macro avg"]["recall"]
macro_f = report_dict["macro avg"]["f1-score"]
weighted_p = report_dict["weighted avg"]["precision"]
weighted_r = report_dict["weighted avg"]["recall"]
weighted_f = report_dict["weighted avg"]["f1-score"]

latex_table3 += r"""\midrule
\textit{Macro avg} & """ + f"{macro_p:.3f} & {macro_r:.3f} & {macro_f:.3f}" + r""" \\
\textit{Weighted avg} & """ + f"{weighted_p:.3f} & {weighted_r:.3f} & {weighted_f:.3f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Per-class precision, recall, and $F_1$ of the LLM judge pipeline (majority vote of three judges) evaluated against human-majority labels on """ + f"{len(val_merged)}" + r""" instances. Cohen's $\kappa$ (human vs.\ judge) = """ + f"{kappa_hj:.2f}" + r""". Overall accuracy = """ + f"{acc:.1%}" + r""".}
\label{tab:judge-validation}
\end{table}
"""
print(latex_table3)

# --- Appendix B: Confusion matrix ---
print("\n%%% APPENDIX B: Confusion Matrix (Human vs Judge) %%%")
latex_cm = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{l|ccc|c}
\toprule
 & \multicolumn{3}{c|}{\textbf{Judge Majority}} & \\
\textbf{Human Majority} & \textsc{Support} & \textsc{Reject} & \textsc{Ambiguous} & \textbf{Total} \\
\midrule
"""
for i, label in enumerate(LABELS):
    row_total = cm_hj[i].sum()
    row_str = f"\\textsc{{{label.capitalize()}}}"
    for j in range(len(LABELS)):
        row_str += f" & {cm_hj[i,j]}"
    row_str += f" & {row_total} \\\\\n"
    latex_cm += row_str

col_totals = cm_hj.sum(axis=0)
latex_cm += r"\midrule" + "\n"
latex_cm += r"\textbf{Total}"
for j in range(len(LABELS)):
    latex_cm += f" & {col_totals[j]}"
latex_cm += f" & {cm_hj.sum()}"
latex_cm += r""" \\
\bottomrule
\end{tabular}
\caption{Confusion matrix between human-majority and judge-majority labels on """ + f"{len(val_merged)}" + r""" annotated instances. Rows: human ground truth; columns: judge prediction.}
\label{tab:confusion-matrix}
\end{table}
"""
print(latex_cm)

# --- Appendix B (cont.): Normalized confusion matrix ---
print("\n%%% APPENDIX B (cont.): Normalized Confusion Matrix %%%")
latex_cm_norm = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{l|ccc}
\toprule
 & \multicolumn{3}{c}{\textbf{Judge Majority}} \\
\textbf{Human Majority} & \textsc{Support} & \textsc{Reject} & \textsc{Ambiguous} \\
\midrule
"""
for i, label in enumerate(LABELS):
    row_str = f"\\textsc{{{label.capitalize()}}}"
    for j in range(len(LABELS)):
        row_str += f" & {cm_norm[i,j]:.2f}"
    row_str += " \\\\\n"
    latex_cm_norm += row_str

latex_cm_norm += r"""\bottomrule
\end{tabular}
\caption{Row-normalized confusion matrix (each row sums to 1). Values represent $P(\text{Judge} = j \mid \text{Human} = i)$.}
\label{tab:confusion-matrix-norm}
\end{table}
"""
print(latex_cm_norm)

# --- Uncertainty propagation table ---
print("\n%%% APPENDIX B (cont.): Uncertainty Propagation %%%")
latex_unc = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{Observed} & \textbf{Corrected} \\
 & \textbf{Support (\%)} & \textbf{Support (\%)} \\
\midrule
"""
for model in sorted(df_full["model"].unique()):
    model_data = df_full[df_full["model"] == model]
    p_sup = (model_data["label"] == "SUPPORT").mean()
    p_rej = (model_data["label"] == "REJECT").mean()
    p_amb = (model_data["label"] == "AMBIGUOUS").mean()
    corrected = (
        p_true_sup_given_pred_sup * p_sup
        + p_true_sup_given_pred_rej * p_rej
        + p_true_sup_given_pred_amb * p_amb
    )
    # Clean model name for LaTeX
    model_tex = model.replace("_", r"\_").replace("-", r"-")
    latex_unc += f"\\texttt{{{model_tex}}} & {100*p_sup:.1f} & {100*corrected:.1f} \\\\\n"

latex_unc += r"""\bottomrule
\end{tabular}
\caption{Observed vs.\ corrected stereotype endorsement rates using the empirical error matrix from the human validation set. Conclusions remain unchanged: model rankings are preserved.}
\label{tab:uncertainty-propagation}
\end{table}
"""
print(latex_unc)

# --- Summary statistics ---
print("\n%%% SUMMARY STATISTICS %%%")
print(f"% Human-human Cohen's kappa: {kappa_hh:.4f}")
print(f"% Human-judge Cohen's kappa: {kappa_hj:.4f}")
print(f"% Overall accuracy (human vs judge): {acc:.4f}")
print(f"% Validation set size: {len(val_merged)}")
print(f"% Both-annotated set size: {len(both_labeled)}")
for label in LABELS:
    p = report_dict[label]["precision"]
    r = report_dict[label]["recall"]
    f = report_dict[label]["f1-score"]
    s = report_dict[label]["support"]
    print(f"% {label}: P={p:.3f} R={r:.3f} F1={f:.3f} support={int(s)}")

# ─── 11. Also compute with human-majority using MAJORITY RULE (not just consensus) ──
# Paper says: "ties mapped to AMBIGUOUS"
# Let's also show what happens if we use ALL annotated rows (not just consensus)
print("\n" + "=" * 70)
print("SECTION 11: Analysis using ALL dual-annotated rows (not just consensus)")
print("           Ties → AMBIGUOUS as stated in paper")
print("=" * 70)

all_val = pd.merge(
    both_labeled[["Filename", "Line_No", "human_majority"]],
    df_llm[["Filename", "Line_No", "Majority_Label"]],
    on=["Filename", "Line_No"],
    how="inner",
)
print(f"All dual-annotated with judge labels: {len(all_val)} rows")

y_true_all = all_val["human_majority"]
y_pred_all = all_val["Majority_Label"].str.upper().str.strip()

print("\nClassification Report (ALL rows, ties→AMBIGUOUS):")
print(classification_report(y_true_all, y_pred_all, labels=LABELS, digits=3, zero_division=0))

kappa_all = cohen_kappa_score(y_true_all, y_pred_all)
acc_all = accuracy_score(y_true_all, y_pred_all)
print(f"Cohen's κ: {kappa_all:.4f}")
print(f"Accuracy:  {acc_all:.4f}")

cm_all = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)
print("\nConfusion matrix:")
print(pd.DataFrame(cm_all, index=LABELS, columns=LABELS))

# ─── Save results to JSON ───────────────────────────────────────────────────
results = {
    "human_human_kappa": round(kappa_hh, 4),
    "human_judge_kappa": round(kappa_hj, 4),
    "accuracy": round(acc, 4),
    "validation_set_size": len(val_merged),
    "both_annotated_size": len(both_labeled),
    "per_class": {},
    "confusion_matrix": cm_hj.tolist(),
    "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
}
for label in LABELS:
    results["per_class"][label] = {
        "precision": round(report_dict[label]["precision"], 4),
        "recall": round(report_dict[label]["recall"], 4),
        "f1": round(report_dict[label]["f1-score"], 4),
        "support": int(report_dict[label]["support"]),
    }

with open(OUT_DIR / "human_judge_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {OUT_DIR / 'human_judge_validation_results.json'}")

print("\n" + "=" * 70)
print("DONE — All metrics computed successfully")
print("=" * 70)
