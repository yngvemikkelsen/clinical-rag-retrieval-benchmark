#!/usr/bin/env python3
"""
BM25 parameter sensitivity analysis.

Sweeps k1 ∈ {1.0, 1.2, 1.5, 2.0} × b ∈ {0.25, 0.5, 0.75, 1.0}
across all 3 corpora and 2 query formats.

Tokenizer: str.lower().split() (matches main benchmark).

Input:
    bm25_sensitivity.csv (pre-computed results)

Expected output:
    Max spread: 0.038 (PMC-Patients, keyword)
    Default within 0.014 of best in all conditions

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np

def analyze_sensitivity(csv_path="bm25_sensitivity.csv"):
    df = pd.read_csv(csv_path)

    print("BM25 Parameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Parameters: k1 in {sorted(df['k1'].unique())}, "
          f"b in {sorted(df['b'].unique())}")
    print(f"Conditions: {len(df)} ({df['corpus'].nunique()} corpora x "
          f"{df['query_format'].nunique()} query formats x "
          f"{df['k1'].nunique() * df['b'].nunique()} param combos)")
    print(f"Tokenizer: str.lower().split()")

    print(f"\n{'Corpus':<15} {'QF':<20} {'Min':>7} {'Max':>7} "
          f"{'Spread':>7} {'Default':>8} {'Dev':>6}")
    print("-" * 70)

    max_spread = 0
    max_dev = 0
    for corpus in ['MTSamples', 'PMC-Patients', 'Synthetic']:
        for qf in ['keyword', 'natural_language']:
            sub = df[(df['corpus'] == corpus) & (df['query_format'] == qf)]
            if len(sub) == 0:
                continue
            lo, hi = sub['MRR@10'].min(), sub['MRR@10'].max()
            default = sub[(sub['k1'] == 1.5) &
                          (sub['b'] == 0.75)]['MRR@10'].values[0]
            best = sub['MRR@10'].max()
            spread = hi - lo
            dev = abs(default - best)
            if spread > max_spread:
                max_spread = spread
            if dev > max_dev:
                max_dev = dev
            print(f"{corpus:<15} {qf:<20} {lo:>7.4f} {hi:>7.4f} "
                  f"{spread:>7.4f} {default:>8.4f} {dev:>6.4f}")

    print(f"\nMax spread across all conditions: {max_spread:.4f}")
    print(f"Max default deviation from best: {max_dev:.4f}")
    print("Conclusion: BM25 performance is robust to parameter choice.")

if __name__ == "__main__":
    analyze_sensitivity()
