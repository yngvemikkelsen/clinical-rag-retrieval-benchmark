# Clinical RAG Retrieval Benchmark

Replication materials for:

**Mikkelsen Y. Clinical Context Variables Collectively Rival Model Choice in Embedding-Based Retrieval: Multi-Corpus Benchmark Study. JMIR Medical Informatics. 2026.**

## Overview

This benchmark evaluates 13 retrieval configurations (10 embedding models, 2 ablation variants, 1 BM25 baseline) across 3 clinical corpora, 2 query formats, and 4 chunking strategies, yielding 294 experimental conditions. A factorial ANOVA decomposes retrieval performance (MRR@10) into contributions from model choice (40.8%), corpus type (24.6%), query format (19.2%), and chunking strategy (0.2%). A per-query mixed-effects analysis (N = 38,500) confirms the factor ordering with proper random effects for query difficulty (ICC = 0.208).

## Repository Structure

```
.
├── clinical_rag_benchmark_v3.py       # Main benchmark script (Colab, GPU required)
├── all_results.csv                    # Full results for 294 conditions
├── metadata_query_results.csv         # Validation experiment results (78 conditions)
├── metadata_queries.json              # Metadata-only queries (validation)
├── mtsamples_sample.csv               # MTSamples corpus (500 documents)
├── pmc_patients_sample.csv            # PMC-Patients corpus (500 documents)
├── synthetic_notes_copy.csv           # Synthetic corpus (500 documents)
├── analysis/
│   ├── paper3_anova_table3.py         # Reproduces Table 3 (ANOVA decomposition)
│   ├── mixed_effects_analysis.py      # Per-query mixed-effects model (BE-3)
│   ├── bm25_sensitivity_analysis.py   # BM25 parameter sensitivity (BE-8)
│   ├── per_query_ranks_embed.csv      # Per-query ranks, embedding models (35,500 rows)
│   ├── per_query_ranks_bm25.csv       # Per-query ranks, BM25 (3,000 rows)
│   └── bm25_sensitivity.csv           # BM25 k1/b sweep results (96 rows)
└── README.md
```

## Models

| Model | HuggingFace ID | Category |
|-------|---------------|----------|
| BioBERT | dmis-lab/biobert-v1.1 | Domain Encoder |
| ClinicalBERT | medicalai/ClinicalBERT | Domain Encoder |
| BioLORD-2023 | FremyCompany/BioLORD-2023 | Biomedical Retriever |
| MedCPT | ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder | Biomedical Retriever (dual) |
| BGE-base | BAAI/bge-base-en-v1.5 | General Embedding |
| GTE-base | thenlper/gte-base | General Embedding |
| Nomic-embed-text | nomic-ai/nomic-embed-text-v1.5 | General Embedding |
| OpenAI-emb3-small | text-embedding-3-small | General API |
| E5-Mistral-7B | intfloat/e5-mistral-7b-instruct | General LLM |
| Phi-3-mini | microsoft/Phi-3-mini-4k-instruct | General LLM |
| E5-Mistral-7B (ablation) | intfloat/e5-mistral-7b-instruct (mean pooling, no instruction) | Ablation |
| Nomic-embed-text (no prefix) | nomic-ai/nomic-embed-text-v1.5 (no prefixes) | Ablation |
| BM25 | rank_bm25 (Okapi, k1=1.5, b=0.75) | Lexical Baseline |

## Corpora

| Corpus | N | Source | Description |
|--------|---|--------|-------------|
| MTSamples | 500 | mtsamples.com | De-identified medical transcriptions, 40 specialties |
| PMC-Patients | 500 | Zhao et al. (2023), HuggingFace | Physician-authored case reports from PubMed Central |
| Synthetic | 500 | Generated (this study) | Mistral-7B-Instruct-v0.2, 20 specialties, temp=0.8, top_p=0.9, seed=42 |

## Reproducing Key Results

### Table 3: ANOVA Decomposition

```bash
pip install pandas statsmodels
python analysis/paper3_anova_table3.py
```

### Mixed-Effects Model

```bash
pip install pandas statsmodels scipy
python analysis/mixed_effects_analysis.py
```

Expected: model chi-squared = 13,362.5 (p < .001), dataset chi-squared = 825.6 (p < .001), query format chi-squared = 4,798.2 (p < .001), ICC = 0.208, pseudo R-squared = 0.376.

### BM25 Sensitivity Analysis

```bash
python analysis/bm25_sensitivity_analysis.py
```

Expected: maximum MRR@10 spread across k1/b combinations = 0.035 (PMC-Patients keyword). Default parameters within 0.012 of best in all conditions.

### Full Benchmark (GPU required)

```bash
# Google Colab with GPU runtime recommended
# Set HF_TOKEN and OPENAI_API_KEY as environment variables
python clinical_rag_benchmark_v3.py
```

Estimated runtime: 2-3 hours on NVIDIA H100.

## Key Findings

1. Model choice is the largest single factor (eta-squared = 0.408), but context variables collectively explain comparable variance (corpus + query format + interactions = 49.0% vs model-related = 47.6%).
2. Model rankings are corpus-dependent (Kendall tau = 0.59 for keyword queries across MTSamples vs PMC-Patients).
3. Query format has large practical effects (BioLORD-2023: MRR@10 from 0.225 to 0.884 on PMC-Patients when switching from keyword to natural language).
4. BM25 dominates on PMC-Patients (MRR@10 = 0.999 for natural language queries) and is robust to parameter tuning (spread at most 0.035).
5. Domain-specific models underperform general-purpose embeddings despite biomedical pretraining.
6. Chunking has minimal impact (eta-squared = 0.002), with maximum effect 12-20% of the model effect.

## Citation

```bibtex
@article{mikkelsen2026embedding,
  title={Clinical Context Variables Collectively Rival Model Choice in
         Embedding-Based Retrieval: Multi-Corpus Benchmark Study},
  author={Mikkelsen, Yngve},
  journal={JMIR Medical Informatics},
  year={2026}
}
```

## License

Code: MIT License. Data: see individual corpus licenses (MTSamples terms of service; PMC-Patients per Zhao et al., 2023; synthetic notes generated for this study).

## Contact

Yngve Mikkelsen, MD MSc DBA

ORCID: 0000-0003-1543-3805
