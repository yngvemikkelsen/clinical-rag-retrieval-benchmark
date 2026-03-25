#!/usr/bin/env python3
"""
Mixed-effects analysis of per-query reciprocal ranks (BE-3).

Fits a linear mixed-effects model:
    reciprocal_rank ~ C(model) + C(dataset) + C(query_format) + (1 | query_id)

Input:
    per_query_ranks_embed.csv  (embedding models, 35,500 rows)
    per_query_ranks_bm25.csv   (BM25, 3,000 rows)

Output:
    Mixed-effects model summary, ICC, likelihood ratio tests.

Requirements:
    pip install pandas numpy statsmodels scipy
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
embed = pd.read_csv("per_query_ranks_embed.csv")
bm25 = pd.read_csv("per_query_ranks_bm25.csv")
df = pd.concat([embed, bm25], ignore_index=True)

print(f"Data: {len(df)} rows, {df['model'].nunique()} models, "
      f"{df.groupby(['model','dataset','query_format']).ngroups} conditions")
print(f"Datasets: {sorted(df['dataset'].unique())}")
print(f"Models: {sorted(df['model'].unique())}")

# Fit full model (REML for variance estimates)
print("\nFitting mixed-effects model (REML)...")
md = smf.mixedlm("reciprocal_rank ~ C(model) + C(dataset) + C(query_format)",
                  data=df, groups=df["query_id"])
mdf = md.fit(reml=True)
print(mdf.summary())

# ICC
re_var = mdf.cov_re.iloc[0, 0]
resid_var = mdf.scale
icc = re_var / (re_var + resid_var)
print(f"\nRandom effect variance (query_id): {re_var:.4f}")
print(f"Residual variance: {resid_var:.4f}")
print(f"ICC(query_id): {icc:.3f}")

# Pseudo-R² (variance reduction vs null model)
md_null = smf.mixedlm("reciprocal_rank ~ 1", data=df, groups=df["query_id"])
mdf_null = md_null.fit(reml=True)
null_total = mdf_null.cov_re.iloc[0, 0] + mdf_null.scale
full_total = re_var + resid_var
pseudo_r2 = 1 - full_total / null_total
print(f"Pseudo R² (variance reduction): {pseudo_r2:.3f}")

# Likelihood ratio tests (ML estimation required)
print("\n=== Likelihood Ratio Tests ===")
md_full_ml = smf.mixedlm("reciprocal_rank ~ C(model) + C(dataset) + C(query_format)",
                           data=df, groups=df["query_id"]).fit(reml=False)

for factor, df_factor in [("C(model)", 12), ("C(dataset)", 2), ("C(query_format)", 1)]:
    reduced_formula = "reciprocal_rank ~ " + " + ".join(
        f for f in ["C(model)", "C(dataset)", "C(query_format)"] if f != factor
    )
    md_reduced = smf.mixedlm(reduced_formula, data=df, groups=df["query_id"]).fit(reml=False)
    chi2 = 2 * (md_full_ml.llf - md_reduced.llf)
    p = 1 - stats.chi2.cdf(chi2, df_factor)
    print(f"  {factor}: χ²={chi2:.1f}, df={df_factor}, p={'<.001' if p < 0.001 else f'{p:.4f}'}")

print("\nDone.")
