"""
Metadata-Only Query Validation Experiment
==========================================
Validates that relative model rankings hold when queries are generated from
document metadata only (specialty, note type, diagnoses) — never seeing the
document text. This addresses the known-item retrieval circularity concern.

Run on Google Colab with H100 GPU runtime.

Prerequisites:
  - Your three corpus JSONL files in data/corpora/
  - OpenAI API key (for GPT-4 metadata extraction + query generation)
  - Original all_results.csv for comparison

Usage:
  1. Upload this script to Colab
  2. Set OPENAI_API_KEY and DATA_DIR paths
  3. Run all cells sequentially
  4. Results saved to metadata_query_results.csv
"""

# %% [markdown]
# # 0. Setup

# %%
# !pip install -q sentence-transformers transformers rank-bm25 openai tiktoken tqdm scipy statsmodels

import os
import json
import re
import time
import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────
# Set these before running:
OPENAI_API_KEY = "sk-..."  # @param {type:"string"}
DATA_DIR = Path("./data")  # Path to your data directory
RESULTS_DIR = Path("./results")
ORIGINAL_RESULTS_CSV = Path("./all_results.csv")  # Your existing 294-condition results

N_DOCS_PER_CORPUS = 100  # 100 per corpus = 300 total (sufficient for validation)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# # 1. Load Corpora and Sample Documents

# %%
def load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file."""
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def sample_documents(docs: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Random sample of n documents."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(docs), size=min(n, len(docs)), replace=False)
    return [docs[i] for i in indices]

# ── Load corpora ──
# ADAPT THESE PATHS to your actual corpus file locations:
corpora = {}
corpus_files = {
    "MTSamples": DATA_DIR / "corpora" / "mtsamples_500.jsonl",
    "PMC-Patients": DATA_DIR / "corpora" / "pmc_patients_500.jsonl",
    "Synthetic": DATA_DIR / "corpora" / "synthetic_500.jsonl",
}

for name, path in corpus_files.items():
    if path.exists():
        all_docs = load_corpus(path)
        corpora[name] = sample_documents(all_docs, N_DOCS_PER_CORPUS, SEED)
        print(f"{name}: loaded {len(all_docs)}, sampled {len(corpora[name])}")
    else:
        print(f"WARNING: {path} not found. Adjust corpus_files paths.")

# %% [markdown]
# # 2. Extract Metadata from Documents (GPT-4)
#
# GPT-4 reads each document and extracts structured metadata.
# This step DOES see the document text — that's fine.
# The key constraint is that QUERY GENERATION (Step 3) sees ONLY metadata.

# %%
from openai import OpenAI

client = OpenAI()

METADATA_EXTRACTION_PROMPT = """Read this clinical document and extract ONLY the following metadata fields.
Return a JSON object with exactly these keys:

{
  "specialty": "<medical specialty, e.g. Cardiology, Orthopedics>",
  "note_type": "<document type, e.g. Operative Report, Consultation Note, Discharge Summary>",
  "primary_diagnosis": "<main diagnosis or chief complaint, 2-5 words>",
  "secondary_diagnoses": ["<up to 2 additional diagnoses, 2-5 words each>"],
  "patient_demographics": "<age range and sex, e.g. elderly male, middle-aged female>"
}

Return ONLY the JSON. No other text.

Document:
{document_text}"""

def extract_metadata(doc_text: str, max_retries: int = 3) -> dict:
    """Extract structured metadata from a document using GPT-4."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-0613",
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "You are a clinical documentation analyst. Return only valid JSON."},
                    {"role": "user", "content": METADATA_EXTRACTION_PROMPT.format(document_text=doc_text[:3000])}
                ]
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {
                    "specialty": "Unknown",
                    "note_type": "Clinical Note",
                    "primary_diagnosis": "Unknown",
                    "secondary_diagnoses": [],
                    "patient_demographics": "adult"
                }

# ── Extract metadata for all sampled documents ──
print("Extracting metadata from documents...")
all_docs_with_metadata = []

for corpus_name, docs in corpora.items():
    print(f"\n{corpus_name} ({len(docs)} docs):")
    for i, doc in enumerate(tqdm(docs, desc=corpus_name)):
        metadata = extract_metadata(doc["text"])
        doc["extracted_metadata"] = metadata
        all_docs_with_metadata.append({
            "doc_id": doc.get("doc_id", f"{corpus_name}_{i}"),
            "corpus": corpus_name,
            "text": doc["text"],
            "metadata": metadata,
        })
        # Rate limiting: ~3 requests/sec to stay under TPM limits
        time.sleep(0.35)

# Save metadata for reproducibility
metadata_path = RESULTS_DIR / "extracted_metadata.jsonl"
with open(metadata_path, "w") as f:
    for doc in all_docs_with_metadata:
        f.write(json.dumps({"doc_id": doc["doc_id"], "corpus": doc["corpus"],
                            "metadata": doc["metadata"]}) + "\n")
print(f"\nSaved metadata to {metadata_path}")

# %% [markdown]
# # 3. Generate Metadata-Only Queries
#
# **CRITICAL**: GPT-4 sees ONLY the metadata fields — never the document text.
# This breaks the query–document lexical overlap that inflates known-item retrieval.

# %%
METADATA_QUERY_PROMPT_NL = """You are a clinician searching for a patient's clinical document.
Based ONLY on the following metadata about the document, write a natural language
clinical question that would help find this document in a search system.

Metadata:
- Specialty: {specialty}
- Note type: {note_type}
- Primary diagnosis: {primary_diagnosis}
- Other conditions: {secondary_diagnoses}
- Patient: {patient_demographics}

Requirements:
1. Write a natural, specific clinical question (1-2 sentences)
2. Use ONLY information from the metadata above
3. Use your medical knowledge to formulate a realistic clinical question
   (you may use synonyms or related clinical terms)
4. Do NOT invent specific lab values, medications, or procedures not implied by the metadata

Query:"""

METADATA_QUERY_PROMPT_KW = """Based ONLY on the following metadata about a clinical document,
generate 3-6 search keywords that a clinician would use to find this document.

Metadata:
- Specialty: {specialty}
- Note type: {note_type}
- Primary diagnosis: {primary_diagnosis}
- Other conditions: {secondary_diagnoses}
- Patient: {patient_demographics}

Requirements:
1. Output only keywords separated by spaces
2. Use ONLY information derivable from the metadata
3. You may use standard medical synonyms
4. 3-6 terms total, no numbering

Keywords:"""

def generate_metadata_query(metadata: dict, query_type: str = "nl",
                            max_retries: int = 3) -> str:
    """Generate a query from metadata only (no document text)."""
    secondary = ", ".join(metadata.get("secondary_diagnoses", [])) or "none"

    if query_type == "nl":
        prompt = METADATA_QUERY_PROMPT_NL.format(
            specialty=metadata.get("specialty", "Unknown"),
            note_type=metadata.get("note_type", "Clinical Note"),
            primary_diagnosis=metadata.get("primary_diagnosis", "Unknown"),
            secondary_diagnoses=secondary,
            patient_demographics=metadata.get("patient_demographics", "adult"),
        )
    else:
        prompt = METADATA_QUERY_PROMPT_KW.format(
            specialty=metadata.get("specialty", "Unknown"),
            note_type=metadata.get("note_type", "Clinical Note"),
            primary_diagnosis=metadata.get("primary_diagnosis", "Unknown"),
            secondary_diagnoses=secondary,
            patient_demographics=metadata.get("patient_demographics", "adult"),
        )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-0613",
                temperature=0.3,  # slight variation for natural phrasing
                max_tokens=150,
                messages=[
                    {"role": "system", "content": "You are a clinical search specialist."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return metadata.get("primary_diagnosis", "clinical query")

# ── Generate queries ──
print("Generating metadata-only queries...")
queries = []

for doc in tqdm(all_docs_with_metadata, desc="Query generation"):
    meta = doc["metadata"]

    nl_query = generate_metadata_query(meta, "nl")
    time.sleep(0.35)
    kw_query = generate_metadata_query(meta, "kw")
    time.sleep(0.35)

    queries.append({
        "doc_id": doc["doc_id"],
        "corpus": doc["corpus"],
        "nl_query": nl_query,
        "kw_query": kw_query,
        "metadata": meta,
    })

# Save queries
queries_path = RESULTS_DIR / "metadata_queries.jsonl"
with open(queries_path, "w") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")
print(f"Saved {len(queries)} query pairs to {queries_path}")

# Diagnostic: check lexical overlap reduction
print("\n── Lexical Overlap Diagnostic ──")
for corpus_name in corpora:
    corpus_queries = [q for q in queries if q["corpus"] == corpus_name]
    corpus_docs = [d for d in all_docs_with_metadata if d["corpus"] == corpus_name]

    overlaps = []
    for q, d in zip(corpus_queries, corpus_docs):
        q_tokens = set(q["kw_query"].lower().split())
        d_tokens = set(d["text"].lower().split())
        jaccard = len(q_tokens & d_tokens) / len(q_tokens | d_tokens) if q_tokens | d_tokens else 0
        overlaps.append(jaccard)
    print(f"  {corpus_name}: mean Jaccard overlap = {np.mean(overlaps):.4f} "
          f"(expect lower than original known-item queries)")

# %% [markdown]
# # 4. Run Retrieval Evaluation
#
# Evaluate all 13 retrieval configurations on metadata-only queries.
# Full-document indexing only (chunking was negligible in main study).

# %%
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Model loading helpers ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "BM25": {"type": "bm25"},
    "BioBERT": {
        "hub": "dmis-lab/biobert-base-cased-v1.2",
        "query_prefix": "", "doc_prefix": "", "pooling": "mean",
    },
    "ClinicalBERT": {
        "hub": "emilyalsentzer/Bio_ClinicalBERT",
        "query_prefix": "", "doc_prefix": "", "pooling": "mean",
    },
    "BioLORD-2023": {
        "hub": "FremyCompany/BioLORD-2023",
        "query_prefix": "", "doc_prefix": "", "pooling": "mean",
    },
    "MedCPT": {
        "hub": "ncbi/MedCPT-Query-Encoder",
        "hub_doc": "ncbi/MedCPT-Article-Encoder",
        "query_prefix": "", "doc_prefix": "", "pooling": "cls",
        "dual_encoder": True,
    },
    "BGE-base": {
        "hub": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "", "pooling": "cls",
    },
    "GTE-base": {
        "hub": "Alibaba-NLP/gte-base-en-v1.5",
        "query_prefix": "", "doc_prefix": "", "pooling": "mean",
    },
    "Nomic-embed-text": {
        "hub": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "search_query: ", "doc_prefix": "search_document: ",
        "pooling": "mean", "trust_remote_code": True,
    },
    "Nomic-nopfx": {
        "hub": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "", "doc_prefix": "",
        "pooling": "mean", "trust_remote_code": True,
    },
    "OpenAI-emb3-small": {"type": "openai", "model": "text-embedding-3-small"},
    "E5-Mistral-7B": {
        "hub": "intfloat/e5-mistral-7b-instruct",
        "query_prefix": "Instruct: Retrieve a clinical document relevant to this query\nQuery: ",
        "doc_prefix": "", "pooling": "last_token",
        "torch_dtype": torch.float16,
    },
    "E5-Mistral-7B-meanpool": {
        "hub": "intfloat/e5-mistral-7b-instruct",
        "query_prefix": "Instruct: Retrieve a clinical document relevant to this query\nQuery: ",
        "doc_prefix": "", "pooling": "mean",
        "torch_dtype": torch.float16,
    },
    "Phi-3-mini": {
        "hub": "microsoft/Phi-3-mini-128k-instruct",
        "query_prefix": "", "doc_prefix": "", "pooling": "mean",
        "torch_dtype": torch.float16,
        "max_length": 4096,
    },
}

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_texts_st(model_name: str, texts: list[str], prefix: str = "",
                    batch_size: int = 32) -> np.ndarray:
    """Encode texts using SentenceTransformers."""
    cfg = MODEL_CONFIGS[model_name]
    prefixed = [prefix + t for t in texts]

    kwargs = {}
    if cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    model = SentenceTransformer(cfg["hub"], device=DEVICE, **kwargs)
    embeddings = model.encode(prefixed, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    del model
    clear_gpu()
    return embeddings


def encode_texts_transformers(model_name: str, texts: list[str], prefix: str = "",
                              batch_size: int = 8) -> np.ndarray:
    """Encode texts using raw transformers (for LLM-based encoders)."""
    from transformers import AutoTokenizer, AutoModel

    cfg = MODEL_CONFIGS[model_name]
    prefixed = [prefix + t for t in texts]

    tokenizer = AutoTokenizer.from_pretrained(cfg["hub"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        cfg["hub"], torch_dtype=cfg.get("torch_dtype", torch.float32)
    ).to(DEVICE).eval()

    max_len = cfg.get("max_length", 4096)
    all_embs = []

    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**encoded)

        if cfg["pooling"] == "last_token":
            # Last non-padding token
            seq_lens = encoded["attention_mask"].sum(dim=1) - 1
            embs = outputs.last_hidden_state[torch.arange(len(batch)), seq_lens]
        else:  # mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1)
            embs = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())

    del model, tokenizer
    clear_gpu()
    return np.vstack(all_embs)


def encode_texts_openai(texts: list[str], model: str = "text-embedding-3-small",
                        batch_size: int = 100) -> np.ndarray:
    """Encode texts using OpenAI API."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embs = [item.embedding for item in response.data]
        all_embs.extend(embs)
        time.sleep(0.5)
    return np.array(all_embs)


def encode_medcpt(queries: list[str], docs: list[str],
                  batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """MedCPT dual encoder: separate query and article encoders."""
    from transformers import AutoTokenizer, AutoModel

    # Query encoder
    q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE).eval()
    q_embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        encoded = q_tok(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = q_model(**encoded).last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        q_embs.append(emb.cpu().numpy())
    del q_model, q_tok
    clear_gpu()

    # Document encoder
    d_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    d_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEVICE).eval()
    d_embs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        encoded = d_tok(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = d_model(**encoded).last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        d_embs.append(emb.cpu().numpy())
    del d_model, d_tok
    clear_gpu()

    return np.vstack(q_embs), np.vstack(d_embs)

# %% [markdown]
# # 4b. Evaluation Loop

# %%
def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize consistently with the main study."""
    import string
    STOPWORDS = {"a","an","the","is","are","was","were","be","been","being","have",
                 "has","had","do","does","did","will","would","shall","should","may",
                 "might","must","can","could","i","me","my","we","our","you","your",
                 "he","him","his","she","her","it","its","they","them","their","this",
                 "that","these","those","in","on","at","to","for","of","with","by",
                 "from","as","into","through","during","before","after","above","below",
                 "between","out","off","over","under","again","further","then","once",
                 "and","but","or","nor","not","so","very","just","about","up","no","if"}
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


def compute_mrr_at_k(rankings: list[int], k: int = 10) -> float:
    """Compute MRR@k from a list of target ranks (1-indexed)."""
    rrs = []
    for rank in rankings:
        if rank <= k:
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)
    return np.mean(rrs)


def compute_hit_rate_at_k(rankings: list[int], k: int = 10) -> float:
    """Compute Hit Rate@k (Recall@k with single relevant doc)."""
    return np.mean([1.0 if r <= k else 0.0 for r in rankings])


def evaluate_model_on_corpus(model_name: str, corpus_name: str,
                             query_type: str,  # "nl" or "kw"
                             corpus_docs: list[dict],
                             corpus_queries: list[dict]) -> dict:
    """Evaluate a single model on a single corpus with one query type."""
    cfg = MODEL_CONFIGS[model_name]

    # Get document texts and queries
    doc_texts = [d["text"] for d in corpus_docs]
    q_key = "nl_query" if query_type == "nl" else "kw_query"
    query_texts = [q[q_key] for q in corpus_queries]

    n_docs = len(doc_texts)
    assert len(query_texts) == n_docs

    if cfg.get("type") == "bm25":
        # BM25 evaluation
        tokenized_corpus = [tokenize_for_bm25(d) for d in doc_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        rankings = []
        for i, q in enumerate(query_texts):
            scores = bm25.get_scores(tokenize_for_bm25(q))
            ranked_indices = np.argsort(-scores)
            rank = int(np.where(ranked_indices == i)[0][0]) + 1
            rankings.append(rank)

    elif cfg.get("type") == "openai":
        # OpenAI API
        q_embs = encode_texts_openai(query_texts, cfg["model"])
        d_embs = encode_texts_openai(doc_texts, cfg["model"])
        sims = q_embs @ d_embs.T
        rankings = []
        for i in range(n_docs):
            ranked = np.argsort(-sims[i])
            rank = int(np.where(ranked == i)[0][0]) + 1
            rankings.append(rank)

    elif model_name == "MedCPT":
        # Dual encoder
        q_embs, d_embs = encode_medcpt(query_texts, doc_texts)
        sims = q_embs @ d_embs.T
        rankings = []
        for i in range(n_docs):
            ranked = np.argsort(-sims[i])
            rank = int(np.where(ranked == i)[0][0]) + 1
            rankings.append(rank)

    elif model_name in ("E5-Mistral-7B", "E5-Mistral-7B-meanpool", "Phi-3-mini"):
        # Transformers-based encoding
        q_embs = encode_texts_transformers(model_name, query_texts,
                                           prefix=cfg["query_prefix"], batch_size=4)
        d_embs = encode_texts_transformers(model_name, doc_texts,
                                           prefix=cfg["doc_prefix"], batch_size=4)
        sims = q_embs @ d_embs.T
        rankings = []
        for i in range(n_docs):
            ranked = np.argsort(-sims[i])
            rank = int(np.where(ranked == i)[0][0]) + 1
            rankings.append(rank)

    else:
        # SentenceTransformers models
        q_embs = encode_texts_st(model_name, query_texts, prefix=cfg["query_prefix"])
        d_embs = encode_texts_st(model_name, doc_texts, prefix=cfg["doc_prefix"])
        sims = q_embs @ d_embs.T
        rankings = []
        for i in range(n_docs):
            ranked = np.argsort(-sims[i])
            rank = int(np.where(ranked == i)[0][0]) + 1
            rankings.append(rank)

    return {
        "model": model_name,
        "corpus": corpus_name,
        "query_type": query_type,
        "query_source": "metadata_only",
        "n_queries": n_docs,
        "MRR@10": compute_mrr_at_k(rankings, 10),
        "HitRate@10": compute_hit_rate_at_k(rankings, 10),
        "HitRate@20": compute_hit_rate_at_k(rankings, 20),
        "MRR@10_CI_lower": np.nan,  # filled below with bootstrap
        "MRR@10_CI_upper": np.nan,
        "median_rank": np.median(rankings),
        "mean_rank": np.mean(rankings),
        "rankings": rankings,
    }


def bootstrap_ci(rankings: list[int], metric_fn, k: int = 10,
                  n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap confidence interval for a ranking metric."""
    rng = np.random.RandomState(SEED)
    boot_values = []
    for _ in range(n_bootstrap):
        sample = rng.choice(rankings, size=len(rankings), replace=True)
        boot_values.append(metric_fn(sample.tolist(), k))
    lower = np.percentile(boot_values, 100 * alpha / 2)
    upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
    return lower, upper

# %%
# ── Run evaluation ────────────────────────────────────────────────────
print("=" * 70)
print("RUNNING METADATA-ONLY QUERY EVALUATION")
print("=" * 70)

all_results = []
model_names = list(MODEL_CONFIGS.keys())

for model_name in model_names:
    print(f"\n{'─' * 50}")
    print(f"Model: {model_name}")
    print(f"{'─' * 50}")

    for corpus_name, docs in corpora.items():
        corpus_queries = [q for q in queries if q["corpus"] == corpus_name]

        for query_type in ["kw", "nl"]:
            qt_label = "keyword" if query_type == "kw" else "NL"
            print(f"  {corpus_name} / {qt_label}...", end=" ", flush=True)

            result = evaluate_model_on_corpus(
                model_name, corpus_name, query_type, docs, corpus_queries
            )

            # Bootstrap CIs
            ci_lo, ci_hi = bootstrap_ci(result["rankings"], compute_mrr_at_k)
            result["MRR@10_CI_lower"] = ci_lo
            result["MRR@10_CI_upper"] = ci_hi

            print(f"MRR@10={result['MRR@10']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

            # Store without raw rankings for CSV
            result_row = {k: v for k, v in result.items() if k != "rankings"}
            all_results.append(result_row)

    clear_gpu()

# %% [markdown]
# # 5. Analysis: Compare With Original Known-Item Results

# %%
# ── Save results ──
results_df = pd.DataFrame(all_results)
results_path = RESULTS_DIR / "metadata_query_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path}")

# ── Load original results for comparison ──
print("\n" + "=" * 70)
print("COMPARISON: Metadata-Only vs Known-Item Retrieval")
print("=" * 70)

if ORIGINAL_RESULTS_CSV.exists():
    orig_df = pd.read_csv(ORIGINAL_RESULTS_CSV)
    # Filter to full-document, matching corpora
    orig_full = orig_df[orig_df["chunk_strategy"] == "full"].copy()

    # Rename for comparison
    orig_full = orig_full.rename(columns={"MRR@10": "MRR@10_original", "dataset": "corpus",
                                          "query_format": "query_type"})
    # Standardize query_type labels
    orig_full["query_type"] = orig_full["query_type"].map(
        lambda x: "kw" if "keyword" in str(x).lower() else "nl"
    )

    # Merge
    compare = results_df.merge(
        orig_full[["model", "corpus", "query_type", "MRR@10_original"]],
        on=["model", "corpus", "query_type"],
        how="left"
    )

    print("\n── Per-Condition Comparison ──")
    print(compare[["model", "corpus", "query_type", "MRR@10", "MRR@10_original"]].to_string(index=False))

    # ── Rank correlation analysis ──
    print("\n── Rank Correlation: Do model rankings hold? ──")
    for corpus_name in corpora:
        for qt in ["kw", "nl"]:
            qt_label = "keyword" if qt == "kw" else "NL"
            sub = compare[(compare["corpus"] == corpus_name) & (compare["query_type"] == qt)]
            sub = sub.dropna(subset=["MRR@10_original"])
            if len(sub) >= 5:
                tau, p_tau = stats.kendalltau(sub["MRR@10"], sub["MRR@10_original"])
                rho, p_rho = stats.spearmanr(sub["MRR@10"], sub["MRR@10_original"])
                print(f"  {corpus_name} / {qt_label}: "
                      f"Kendall τ = {tau:.3f} (p={p_tau:.3f}), "
                      f"Spearman ρ = {rho:.3f} (p={p_rho:.3f})")

    # ── BM25 comparison: did its advantage shrink? ──
    print("\n── BM25 Performance: Known-Item vs Metadata-Only ──")
    bm25_results = compare[compare["model"] == "BM25"]
    for _, row in bm25_results.iterrows():
        delta = row["MRR@10"] - row["MRR@10_original"] if pd.notna(row["MRR@10_original"]) else float("nan")
        print(f"  {row['corpus']}/{row['query_type']}: "
              f"metadata={row['MRR@10']:.3f}, original={row.get('MRR@10_original', 'N/A')}, "
              f"Δ={delta:+.3f}" if pd.notna(delta) else "  N/A")

    # ── Overall effect: is absolute performance lower? ──
    print("\n── Overall Performance Drop (metadata vs known-item) ──")
    merged = compare.dropna(subset=["MRR@10_original"])
    if len(merged) > 0:
        mean_meta = merged["MRR@10"].mean()
        mean_orig = merged["MRR@10_original"].mean()
        print(f"  Mean MRR@10: metadata={mean_meta:.3f}, original={mean_orig:.3f}, "
              f"Δ={mean_meta - mean_orig:+.3f}")
        print(f"  Expected: absolute drop (confirms non-circularity)")
        print(f"  Key question: do rank correlations remain high?")
else:
    print(f"Original results CSV not found at {ORIGINAL_RESULTS_CSV}")
    print("Skipping comparison analysis. Run independently or adjust path.")

# %% [markdown]
# # 6. Summary for Manuscript

# %%
print("\n" + "=" * 70)
print("MANUSCRIPT TEXT SUGGESTION")
print("=" * 70)
print("""
To add to section 4 (Results) as a new subsection, e.g. "4.9. Validation With
Non-Derived Queries":

  To validate that relative model rankings are not artifacts of the known-item
  retrieval design, we repeated the evaluation using metadata-only queries for
  a subset of {n} documents per corpus. Queries were generated by GPT-4 from
  structured metadata (specialty, note type, primary diagnosis, patient
  demographics) without access to document text. As expected, absolute
  performance was lower for all models (mean MRR@10 drop of X.XXX), confirming
  that the known-item design inflates retrieval scores. Critically, model
  rankings remained [stable/largely stable]: Kendall τ between known-item
  and metadata-only rankings was X.XX–X.XX across corpora and query types,
  and BM25's advantage on PMC-Patients [persisted but was reduced / diminished
  substantially], with MRR@10 dropping from 0.999 to X.XXX for natural
  language queries.

Fill in values from the analysis above.
""".format(n=N_DOCS_PER_CORPUS))

print("Done! Check results/ directory for all outputs.")
