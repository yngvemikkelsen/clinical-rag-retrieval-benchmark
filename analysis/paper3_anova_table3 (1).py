#!/usr/bin/env python3
"""
Paper 3 (JMIR): Factorial ANOVA — Table 3 Reproduction
=======================================================
Reproduces the variance decomposition of MRR@10 across 288 balanced
embedding conditions (12 models × 3 corpora × 2 query formats × 4 chunking).

Input:  all_results.csv from clinical_rag_benchmark_v4.2
Output: Table 3 values + sensitivity analysis

Requires: pandas, statsmodels, numpy
"""

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def run_primary_anova(df):
    """Primary analysis: 288 balanced embedding conditions, Type II SS."""
    embed = df[df['model'] != 'BM25'].copy()
    embed = embed.rename(columns={'query_format': 'qf', 'chunk_strategy': 'chunk'})
    
    assert len(embed) == 288, f"Expected 288 rows, got {len(embed)}"
    
    formula = ('Q("MRR@10") ~ C(model) + C(dataset) + C(qf) + C(chunk) '
               '+ C(model):C(dataset) + C(model):C(qf) + C(model):C(chunk) '
               '+ C(dataset):C(qf) + C(dataset):C(chunk) + C(qf):C(chunk)')
    
    fit = ols(formula, data=embed).fit()
    table = anova_lm(fit, typ=2)
    
    ss_total = table['sum_sq'].sum()
    table['eta_sq'] = table['sum_sq'] / ss_total
    
    return table, fit.rsquared

def run_sensitivity_anova(df):
    """Sensitivity analysis: full-document only, including BM25 (N=78)."""
    full = df[df['chunk_strategy'] == 'full'].copy()
    full = full.rename(columns={'query_format': 'qf'})
    
    assert len(full) == 78, f"Expected 78 rows, got {len(full)}"
    
    formula = 'Q("MRR@10") ~ C(model) + C(dataset) + C(qf)'
    fit = ols(formula, data=full).fit()
    table = anova_lm(fit, typ=2)
    
    ss_total = table['sum_sq'].sum()
    table['eta_sq'] = table['sum_sq'] / ss_total
    
    return table, fit.rsquared

if __name__ == "__main__":
    df = pd.read_csv("all_results.csv")
    
    print("=" * 70)
    print("PRIMARY ANOVA (N=288, Type II, all two-way interactions)")
    print("=" * 70)
    table, r2 = run_primary_anova(df)
    
    names = {
        'C(model)': 'Embedding model', 'C(dataset)': 'Corpus',
        'C(qf)': 'Query format', 'C(chunk)': 'Chunking strategy',
        'C(model):C(dataset)': 'Model × Corpus',
        'C(model):C(qf)': 'Model × Query format',
        'C(model):C(chunk)': 'Model × Chunking',
        'C(dataset):C(qf)': 'Corpus × Query format',
        'C(dataset):C(chunk)': 'Corpus × Chunking',
        'C(qf):C(chunk)': 'QF × Chunking',
        'Residual': 'Residual',
    }
    
    print(f"\n{'Factor':<30} {'df':>4} {'η²':>7} {'F':>10} {'p':>10}")
    print("-" * 65)
    for idx, row in table.iterrows():
        name = names.get(str(idx), str(idx))
        p = row.get('PR(>F)', float('nan'))
        p_str = '< .001' if (not pd.isna(p) and p < 0.001) else f'{p:.4f}' if not pd.isna(p) else ''
        F_str = f'{row["F"]:.2f}' if not pd.isna(row.get('F', float('nan'))) else ''
        print(f"{name:<30} {row.get('df',0):>4.0f} {row['eta_sq']:>7.3f} {F_str:>10} {p_str:>10}")
    print(f"\nR² = {r2:.4f}")
    
    print(f"\n{'=' * 70}")
    print("SENSITIVITY ANALYSIS (N=78, full-doc + BM25, main effects only)")
    print("=" * 70)
    table2, r2_2 = run_sensitivity_anova(df)
    for idx, row in table2.iterrows():
        name = names.get(str(idx), str(idx))
        print(f"  {name:<25} η² = {row['eta_sq']:.3f} ({row['eta_sq']*100:.1f}%)")
    print(f"  R² = {r2_2:.3f}")
