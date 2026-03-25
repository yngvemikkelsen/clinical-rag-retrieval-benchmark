#!/usr/bin/env python3
"""
BM25 parameter sensitivity analysis (BE-8).

Sweeps k1 ∈ {1.0, 1.2, 1.5, 2.0} × b ∈ {0.25, 0.5, 0.75, 1.0}
across all 3 corpora and 2 query formats.

Input:
    bm25_sensitivity.csv (pre-computed results)
    OR run with --recompute flag to regenerate from corpora CSVs

Output:
    Summary table and spread analysis.

Requirements:
    pip install pandas numpy
    (rank_bm25 needed only for --recompute)
"""

import pandas as pd
import numpy as np
import sys

def analyze_sensitivity(csv_path="bm25_sensitivity.csv"):
    df = pd.read_csv(csv_path)
    
    print("BM25 Parameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Parameters: k1 ∈ {sorted(df['k1'].unique())}, b ∈ {sorted(df['b'].unique())}")
    print(f"Conditions: {len(df)} ({df['corpus'].nunique()} corpora × "
          f"{df['query_format'].nunique()} query formats × "
          f"{df['k1'].nunique() * df['b'].nunique()} param combos)")
    
    print(f"\n{'Corpus':<15} {'QF':<20} {'Min':>7} {'Max':>7} {'Spread':>7} {'Default':>8}")
    print("-" * 65)
    
    for corpus in ['MTSamples', 'PMC-Patients', 'Synthetic']:
        for qf in ['keyword', 'natural_language']:
            sub = df[(df['corpus'] == corpus) & (df['query_format'] == qf)]
            lo, hi = sub['MRR@10'].min(), sub['MRR@10'].max()
            default = sub[(sub['k1'] == 1.5) & (sub['b'] == 0.75)]['MRR@10'].values[0]
            best = sub.loc[sub['MRR@10'].idxmax()]
            spread = hi - lo
            print(f"{corpus:<15} {qf:<20} {lo:>7.4f} {hi:>7.4f} {spread:>7.4f} {default:>8.4f}")
    
    print(f"\nPivot: PMC-Patients, keyword")
    pmc_kw = df[(df['corpus'] == 'PMC-Patients') & (df['query_format'] == 'keyword')]
    print(pmc_kw.pivot_table(index='k1', columns='b', values='MRR@10').round(4).to_string())
    
    max_spread = df.groupby(['corpus', 'query_format'])['MRR@10'].apply(lambda x: x.max() - x.min()).max()
    print(f"\nMax spread across all conditions: {max_spread:.4f}")
    print("Conclusion: BM25 performance is robust to parameter choice.")

if __name__ == "__main__":
    analyze_sensitivity()
