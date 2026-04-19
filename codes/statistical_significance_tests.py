#!/usr/bin/env python3
"""
Statistical Significance Tests for Mental Health Bias Paper (Appendix C)

This script performs:
1. Bootstrap confidence intervals for model endorsement rates
2. Pairwise model comparisons using McNemar's test
3. Chi-square tests for DSM-5 category differences
4. Generates LaTeX tables for paper appendix

Author: Research Team
Date: February 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency
try:
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
    HAS_MCNEMAR = True
except ImportError:
    HAS_MCNEMAR = False
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# Configuration
BASE_DIR = Path("/home/ad2688/Research/Works_Done/mental_disorder")
MV_DIR = BASE_DIR / "results_new" / "majority_vote"
DISORDERS_FILE = BASE_DIR / "filtered_combined_disorders.json"
OUTPUT_DIR = BASE_DIR
N_BOOTSTRAP = 5000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def bootstrap_ci(data, n_iterations=5000, confidence_level=0.95, metric_fn=np.mean):
    """
    Calculate bootstrap confidence interval for a given metric.
    
    Parameters:
    - data: array-like of binary outcomes (1=SUPPORT, 0=not SUPPORT)
    - n_iterations: number of bootstrap samples
    - confidence_level: confidence level (default 0.95 for 95% CI)
    - metric_fn: function to compute metric (default: mean for rates)
    
    Returns:
    - Dictionary with point_estimate, ci_lower, ci_upper
    """
    n = len(data)
    bootstrap_estimates = []
    
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_estimates.append(metric_fn(sample))
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(data)
    
    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower
    }


def load_model_data():
    """Load all majority vote files and return structured data."""
    print("=" * 70)
    print("Loading model data...")
    print("=" * 70)
    
    mv_files = sorted(MV_DIR.glob("majority_vote_*.jsonl"))
    print(f"Found {len(mv_files)} majority vote files\n")
    
    model_data = {}
    for mv_file in mv_files:
        model_name = mv_file.stem.replace("majority_vote_", "")
        
        # Load data
        data = []
        with open(mv_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        # Create structured predictions
        model_predictions = {}
        for entry in data:
            key = (
                entry.get('group', '').strip().lower(),
                entry.get('stereotype', '').strip(),
                entry.get('line_no', 0)
            )
            # Extract label from majority_vote dictionary
            mv_dict = entry.get('majority_vote', {})
            if isinstance(mv_dict, dict):
                label_value = mv_dict.get('label', '')
            else:
                # Fallback if it's a string
                label_value = str(mv_dict) if mv_dict else ''
            
            label = 1 if label_value == 'SUPPORT' else 0
            model_predictions[key] = label
        
        model_data[model_name] = {
            'predictions': model_predictions,
            'raw_data': data
        }
        
        # Calculate support rate
        support_count = sum(1 for v in model_predictions.values() if v == 1)
        total = len(model_predictions)
        support_rate = support_count / total if total > 0 else 0
        
        print(f"  {model_name}: {len(data)} instances, {support_count} SUPPORT ({support_rate*100:.1f}%)")
    
    return model_data


def compute_bootstrap_cis(model_data):
    """Compute bootstrap confidence intervals for all models."""
    print("\n" + "=" * 70)
    print("SECTION 1: Bootstrap Confidence Intervals")
    print("=" * 70)
    
    bootstrap_results = {}
    for model_name, data_dict in model_data.items():
        # Extract SUPPORT labels from raw data
        support_labels = []
        for entry in data_dict['raw_data']:
            mv_dict = entry.get('majority_vote', {})
            if isinstance(mv_dict, dict):
                label_value = mv_dict.get('label', '')
            else:
                label_value = str(mv_dict) if mv_dict else ''
            
            support_labels.append(1 if label_value == 'SUPPORT' else 0)
        
        if not support_labels:
            continue
        
        # Compute bootstrap CI
        ci_result = bootstrap_ci(support_labels, n_iterations=N_BOOTSTRAP, confidence_level=0.95)
        bootstrap_results[model_name] = ci_result
        
        print(f"\n{model_name}:")
        print(f"  Support rate: {ci_result['point_estimate']:.4f} ({ci_result['point_estimate']*100:.2f}%)")
        print(f"  95% CI: [{ci_result['ci_lower']*100:.2f}%, {ci_result['ci_upper']*100:.2f}%]")
        print(f"  CI width: {ci_result['ci_width']*100:.2f} pp")
    
    print(f"\n✓ Bootstrap CIs computed for {len(bootstrap_results)} models")
    return bootstrap_results


def perform_mcnemar_tests(model_data):
    """Perform pairwise McNemar tests between all model pairs."""
    print("\n" + "=" * 70)
    print("SECTION 2: Pairwise Model Comparisons (McNemar's Test)")
    print("=" * 70)
    
    model_names = sorted(model_data.keys())
    print(f"\nModels: {len(model_names)}")
    print(f"Total pairwise comparisons: {len(list(combinations(model_names, 2)))}\n")
    
    mcnemar_results = []
    
    for model1, model2 in combinations(model_names, 2):
        # Get predictions
        pred1 = model_data[model1]['predictions']
        pred2 = model_data[model2]['predictions']
        
        # Find common instances
        common_keys = set(pred1.keys()) & set(pred2.keys())
        
        if len(common_keys) < 10:
            continue
        
        # Build contingency table
        a = b = c = d = 0
        for key in common_keys:
            p1 = pred1[key]
            p2 = pred2[key]
            
            if p1 == 1 and p2 == 1:
                a += 1
            elif p1 == 1 and p2 == 0:
                b += 1
            elif p1 == 0 and p2 == 1:
                c += 1
            else:
                d += 1
        
        # McNemar test - manual implementation
        # Test statistic: (|b - c| - 1)^2 / (b + c) with continuity correction
        if b + c < 25:
            # Exact binomial test
            p_value = 2 * min(
                stats.binom.cdf(min(b, c), b + c, 0.5),
                1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5)
            )
            test_type = "exact"
        else:
            # Chi-square approximation with continuity correction
            test_statistic = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(test_statistic, df=1)
            test_type = "chi-square"
        
        # Compute rate difference
        rate1 = (a + b) / (a + b + c + d)
        rate2 = (a + c) / (a + b + c + d)
        
        mcnemar_results.append({
            'model1': model1,
            'model2': model2,
            'n_common': len(common_keys),
            'model1_support_rate': rate1,
            'model2_support_rate': rate2,
            'rate_difference': rate1 - rate2,
            'a': a, 'b': b, 'c': c, 'd': d,
            'discordant_pairs': b + c,
            'p_value': p_value,
            'test_type': test_type,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01
        })
    
    mcnemar_df = pd.DataFrame(mcnemar_results)
    
    print(f"✓ Completed {len(mcnemar_df)} pairwise comparisons")
    print(f"\nSignificant at α=0.05: {mcnemar_df['significant_at_05'].sum()} ({mcnemar_df['significant_at_05'].mean()*100:.1f}%)")
    print(f"Significant at α=0.01: {mcnemar_df['significant_at_01'].sum()} ({mcnemar_df['significant_at_01'].mean()*100:.1f}%)")
    print(f"Median p-value: {mcnemar_df['p_value'].median():.4f}")
    
    # Show top significant differences
    print("\n" + "=" * 70)
    print("Top 10 Most Significant Differences:")
    print("=" * 70)
    sig_comparisons = mcnemar_df.nsmallest(10, 'p_value')
    for idx, row in sig_comparisons.iterrows():
        print(f"\n{row['model1']} vs {row['model2']}:")
        print(f"  Rates: {row['model1_support_rate']:.3f} vs {row['model2_support_rate']:.3f}")
        print(f"  Difference: {row['rate_difference']:+.3f} ({row['rate_difference']*100:+.1f} pp)")
        print(f"  p-value: {row['p_value']:.2e}")
    
    return mcnemar_df


def perform_dsm5_chi_square_tests(model_data):
    """Perform chi-square tests for DSM-5 category differences."""
    print("\n" + "=" * 70)
    print("SECTION 3: DSM-5 Category Differences (Chi-Square Tests)")
    print("=" * 70)
    
    # Load DSM-5 mappings - this is just a list of disorder names
    # We need to load the actual DSM-5 category mappings from another source
    # For now, let's create a simplified mapping based on common knowledge
    
    # Simple DSM-5 category mapping (subset of major categories)
    dsm5_categories = {
        'Anxiety Disorders': ['agoraphobia', 'panic disorder', 'social anxiety', 'generalized anxiety', 
                              'specific phobia', 'selective mutism'],
        'Depressive Disorders': ['major depression', 'dysthymia', 'persistent depressive', 
                                 'seasonal affective', 'postpartum depression'],
        'Bipolar Disorders': ['bipolar', 'cyclothymia', 'hypomania'],
        'Schizophrenia Spectrum': ['schizophrenia', 'schizoaffective', 'delusional disorder', 
                                   'psychotic disorder', 'schizotypal', 'schizoid'],
        'Trauma-Related Disorders': ['ptsd', 'post-traumatic', 'acute stress', 'adjustment disorder'],
        'OCD Related': ['ocd', 'obsessive-compulsive', 'hoarding', 'trichotillomania', 
                        'body dysmorphic', 'excoriation'],
        'Eating Disorders': ['anorexia', 'bulimia', 'binge eating', 'avoidant restrictive food intake'],
        'Personality Disorders': ['borderline', 'antisocial', 'narcissistic', 'paranoid personality',
                                  'schizoid personality', 'avoidant personality', 'dependent personality',
                                  'obsessive-compulsive personality'],
        'Neurodevelopmental': ['adhd', 'autism', 'intellectual disability', 'learning disorder', 
                               'language disorder', 'developmental coordination'],
        'Substance-Related': ['alcohol', 'substance', 'addiction', 'dependence'],
        'Dissociative Disorders': ['dissociative identity', 'dissociative amnesia', 'depersonalization',
                                   'derealization', 'ganser syndrome'],
        'Somatic Disorders': ['somatic symptom', 'illness anxiety', 'conversion', 'factitious'],
    }
    
    print(f"\nDSM-5 categories defined: {len(dsm5_categories)}\n")
    
    chi_square_results = []
    
    for model_name, data_dict in model_data.items():
        predictions = data_dict['predictions']
        
        # Count by category
        category_counts = {cat: {'support': 0, 'total': 0} for cat in dsm5_categories}
        unmatched = 0
        
        for key, label in predictions.items():
            group = key[0]  # Already lowercased
            
            # Find category (match if any keyword appears in group name)
            found_category = None
            for cat, keywords in dsm5_categories.items():
                if any(keyword in group for keyword in keywords):
                    found_category = cat
                    break
            
            if found_category:
                category_counts[found_category]['total'] += 1
                if label == 1:
                    category_counts[found_category]['support'] += 1
            else:
                unmatched += 1
        
        # Build contingency table
        categories_with_data = [
            (cat, counts) for cat, counts in category_counts.items()
            if counts['total'] >= 5  # Minimum 5 instances per category
        ]
        
        if len(categories_with_data) < 2:
            print(f"{model_name}: Insufficient categories with data (need ≥2, have {len(categories_with_data)})")
            continue
        
        support_counts = [counts['support'] for _, counts in categories_with_data]
        non_support_counts = [counts['total'] - counts['support'] for _, counts in categories_with_data]
        
        contingency_table = np.array([support_counts, non_support_counts])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        chi_square_results.append({
            'model': model_name,
            'n_categories': len(categories_with_data),
            'unmatched': unmatched,
            'chi2': chi2,
            'dof': dof,
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01
        })
        
        print(f"{model_name}:")
        print(f"  Categories: {len(categories_with_data)}, χ²={chi2:.2f}, p={p_value:.4e}")
    
    chi_square_df = pd.DataFrame(chi_square_results)
    
    print(f"\n✓ Chi-square tests completed for {len(chi_square_df)} models")
    print(f"Significant at α=0.05: {chi_square_df['significant_at_05'].sum()} / {len(chi_square_df)}")
    
    return chi_square_df


def generate_latex_tables(bootstrap_results, mcnemar_df, chi_square_df):
    """Generate LaTeX tables for Appendix C."""
    print("\n" + "=" * 70)
    print("SECTION 4: LaTeX Tables for Appendix C")
    print("=" * 70)
    
    # Table 1: Bootstrap CIs
    print("\n" + "=" * 70)
    print("TABLE 1: Bootstrap Confidence Intervals")
    print("=" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Support Rate} & \textbf{95\% CI} & \textbf{Width} \\")
    print(r"\midrule")
    
    sorted_models = sorted(bootstrap_results.items(), key=lambda x: x[1]['point_estimate'], reverse=True)
    for model_name, ci_result in sorted_models:
        rate = ci_result['point_estimate'] * 100
        ci_lower = ci_result['ci_lower'] * 100
        ci_upper = ci_result['ci_upper'] * 100
        ci_width = ci_result['ci_width'] * 100
        
        clean_name = model_name.replace('_', r'\_')
        print(f"\\texttt{{{clean_name}}} & {rate:.2f}\\% & [{ci_lower:.2f}, {ci_upper:.2f}] & {ci_width:.2f}pp \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Bootstrap confidence intervals (5,000 iterations) for model endorsement rates.}")
    print(r"\label{tab:bootstrap-ci}")
    print(r"\end{table}")
    
    # Table 2: McNemar Tests (Top 20)
    print("\n" + "=" * 70)
    print("TABLE 2: Pairwise Model Comparisons (McNemar's Test)")
    print("=" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\begin{tabular}{llcccp{1.5cm}}")
    print(r"\toprule")
    print(r"\textbf{Model 1} & \textbf{Model 2} & \textbf{Rate 1} & \textbf{Rate 2} & \textbf{Diff} & \textbf{$p$-value} \\")
    print(r"\midrule")
    
    top_comparisons = mcnemar_df.nsmallest(20, 'p_value')
    for idx, row in top_comparisons.iterrows():
        m1 = row['model1'].replace('_', r'\_')[:25]
        m2 = row['model2'].replace('_', r'\_')[:25]
        rate1 = row['model1_support_rate'] * 100
        rate2 = row['model2_support_rate'] * 100
        diff = row['rate_difference'] * 100
        pval = row['p_value']
        
        if pval < 0.0001:
            pval_str = r"$<10^{-4}$"
        else:
            pval_str = f"{pval:.4f}"
        
        print(f"\\texttt{{{m1}}} & \\texttt{{{m2}}} & {rate1:.1f}\\% & {rate2:.1f}\\% & {diff:+.1f}pp & {pval_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Pairwise model comparisons using McNemar's test (top 20 by significance).}")
    print(r"\label{tab:mcnemar-test}")
    print(r"\end{table}")
    
    # Table 3: Chi-Square Tests
    print("\n" + "=" * 70)
    print("TABLE 3: Chi-Square Tests for DSM-5 Category Differences")
    print("=" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcccp{2cm}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Categories} & \textbf{$\chi^2$} & \textbf{DoF} & \textbf{$p$-value} \\")
    print(r"\midrule")
    
    for idx, row in chi_square_df.iterrows():
        model = row['model'].replace('_', r'\_')[:30]
        n_cat = row['n_categories']
        chi2 = row['chi2']
        dof = row['dof']
        pval = row['p_value']
        
        if pval < 0.0001:
            pval_str = r"$<10^{-4}$"
        else:
            pval_str = f"{pval:.4e}"
        
        sig_marker = r"$^{*}$" if row['significant_at_05'] else ""
        
        print(f"\\texttt{{{model}}} & {n_cat} & {chi2:.2f} & {dof} & {pval_str}{sig_marker} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Chi-square tests for endorsement rate differences across DSM-5 categories. $^{*}p<0.05$.}")
    print(r"\label{tab:dsm5-chisquare}")
    print(r"\end{table}")
    
    print("\n✓ All LaTeX tables generated")


def save_results(bootstrap_results, mcnemar_df, chi_square_df):
    """Save results to JSON and CSV files."""
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # Save bootstrap results
    bootstrap_output = {
        model: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                for k, v in results.items()}
        for model, results in bootstrap_results.items()
    }
    
    with open(OUTPUT_DIR / "bootstrap_ci_results.json", 'w') as f:
        json.dump(bootstrap_output, f, indent=2)
    print(f"✓ Saved: {OUTPUT_DIR / 'bootstrap_ci_results.json'}")
    
    # Save McNemar results
    mcnemar_df.to_csv(OUTPUT_DIR / "mcnemar_pairwise_tests.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'mcnemar_pairwise_tests.csv'}")
    
    # Save chi-square results
    chi_square_df.to_csv(OUTPUT_DIR / "dsm5_chi_square_tests.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'dsm5_chi_square_tests.csv'}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS FOR MENTAL HEALTH BIAS PAPER")
    print("Appendix C: Statistical Testing")
    print("=" * 80)
    
    # Load data
    model_data = load_model_data()
    
    # Perform tests
    bootstrap_results = compute_bootstrap_cis(model_data)
    mcnemar_df = perform_mcnemar_tests(model_data)
    chi_square_df = perform_dsm5_chi_square_tests(model_data)
    
    # Generate tables
    generate_latex_tables(bootstrap_results, mcnemar_df, chi_square_df)
    
    # Save results
    save_results(bootstrap_results, mcnemar_df, chi_square_df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING COMPLETE")
    print("=" * 80)
    print(f"\nTests performed:")
    print(f"  1. Bootstrap CIs: {len(bootstrap_results)} models")
    print(f"  2. McNemar's tests: {len(mcnemar_df)} pairwise comparisons")
    print(f"  3. Chi-square tests: {len(chi_square_df)} models × DSM-5 categories")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Ready for insertion into Appendix C of the paper.\n")


if __name__ == "__main__":
    main()
