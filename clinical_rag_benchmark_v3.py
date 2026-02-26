#!/usr/bin/env python3
"""
Clinical RAG Embedding Benchmark v3
====================================
11 Models × 3 Datasets × 2 Query Formats

Datasets:
  1. MTSamples — medical transcriptions (Kaggle/HF)
  2. PMC-Patients — physician-authored case reports (Zhao et al., Sci Data 2023)
  3. Synthetic — LLM-generated clinical notes (Mistral-7B-Instruct, reproducible)

Run on Google Colab with H100 GPU runtime.
Estimated wall time: 2-3 hours (including synthetic generation).

Usage:
    1. Upload to Google Colab
    2. Set runtime to GPU (H100 preferred, A100/T4 also work)
    3. Set HF_TOKEN and OPENAI_API_KEY in CONFIG section
    4. Run all cells
"""

# ============================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================
# !pip install -q transformers sentence-transformers datasets openai huggingface_hub
# !pip install -q torch  # usually pre-installed on Colab
# !pip install -q scipy scikit-learn pandas numpy matplotlib seaborn tqdm accelerate

# ============================================================
# CELL 2: CONFIG
# ============================================================
import os
import json
import datetime
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── USER CONFIG ──────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # set via Colab secrets or paste here
HF_TOKEN = os.environ.get("HF_TOKEN", "")              # set via Colab secrets or paste here
N_SAMPLES = 500                 # samples per dataset
N_BOOTSTRAP = 1000              # bootstrap resamples for CIs
SEED = 42
OUTPUT_DIR = Path("./benchmark_v3_outputs")
SAVE_EMBEDDINGS = True          # save .npy embedding matrices
SYNTH_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # for synthetic note generation
# ─────────────────────────────────────────────────────────────

# ── Authenticate HuggingFace (for gated models) ──
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("HuggingFace authenticated ✓")
    except Exception as e:
        print(f"HF login warning: {e} — continuing without auth")
else:
    print("⚠ No HF_TOKEN set. Some gated models may fail to download.")
    print("  Set HF_TOKEN in Colab Secrets or paste above.")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "embeddings").mkdir(exist_ok=True)
(OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: MODEL REGISTRY
# ============================================================
@dataclass
class ModelConfig:
    name: str
    hf_id: str
    category: str
    pooling: str = "mean"          # mean | cls | eos
    query_prefix: str = ""
    doc_prefix: str = ""
    instruction: str = ""          # for E5-Mistral
    is_api: bool = False
    is_dual_encoder: bool = False  # for MedCPT
    hf_id_query: str = ""          # for MedCPT
    hf_id_doc: str = ""            # for MedCPT
    dtype: str = "fp32"            # fp32 | fp16
    max_length: int = 512
    use_sentence_transformers: bool = False

MODELS: List[ModelConfig] = [
    # ── Domain Encoders ──
    ModelConfig(
        name="BioBERT",
        hf_id="dmis-lab/biobert-v1.1",
        category="Domain Encoder",
    ),
    ModelConfig(
        name="ClinicalBERT",
        hf_id="medicalai/ClinicalBERT",
        category="Domain Encoder",
    ),
    # ── Biomedical Retrievers ──
    ModelConfig(
        name="BioLORD-2023",
        hf_id="FremyCompany/BioLORD-2023",
        category="Biomedical Retriever",
    ),
    ModelConfig(
        name="MedCPT",
        hf_id="ncbi/MedCPT-Query-Encoder",
        category="Biomedical Retriever",
        pooling="cls",
        is_dual_encoder=True,
        hf_id_query="ncbi/MedCPT-Query-Encoder",
        hf_id_doc="ncbi/MedCPT-Article-Encoder",
    ),
    # ── General Embeddings ──
    ModelConfig(
        name="BGE-base",
        hf_id="BAAI/bge-base-en-v1.5",
        category="General Embedding",
    ),
    ModelConfig(
        name="GTE-base",
        hf_id="thenlper/gte-base",
        category="General Embedding",
    ),
    ModelConfig(
        name="Nomic-embed-text",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        category="General Embedding",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        use_sentence_transformers=True,
    ),
    # ── General API ──
    ModelConfig(
        name="OpenAI-emb3-small",
        hf_id="text-embedding-3-small",
        category="General API",
        is_api=True,
    ),
    # ── General LLMs ──
    ModelConfig(
        name="E5-Mistral-7B",
        hf_id="intfloat/e5-mistral-7b-instruct",
        category="General LLM",
        pooling="eos",
        instruction="Given a clinical note, retrieve the most relevant clinical document.",
        dtype="fp16",
        max_length=4096,
    ),
    ModelConfig(
        name="Phi-3-mini",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        category="General LLM",
        pooling="mean",
        dtype="fp16",
        max_length=4096,
    ),
]

# ── Ablations ──
MODELS.append(ModelConfig(
    name="E5-Mistral-7B-ablation",
    hf_id="intfloat/e5-mistral-7b-instruct",
    category="General LLM",
    pooling="mean",
    instruction="",
    dtype="fp16",
    max_length=4096,
))
MODELS.append(ModelConfig(
    name="Nomic-embed-text-nopfx",
    hf_id="nomic-ai/nomic-embed-text-v1.5",
    category="General Embedding",
    query_prefix="",
    doc_prefix="",
    use_sentence_transformers=True,
))

print(f"Total model configs: {len(MODELS)} ({len(MODELS)-2} primary + 2 ablations)")


# ============================================================
# CELL 4: DATA LOADING — MTSAMPLES
# ============================================================
def load_mtsamples(n=500, seed=42):
    """Load and sample MTSamples clinical notes from HuggingFace."""
    from datasets import load_dataset

    print("Loading MTSamples...")
    try:
        ds = load_dataset("mteb/mtsamples", split="train", token=HF_TOKEN or None)
        df = ds.to_pandas()
    except Exception:
        try:
            ds = load_dataset("lewtun/mtsamples", split="train", token=HF_TOKEN or None)
            df = ds.to_pandas()
        except Exception:
            # Fallback: try local CSV
            for p in ["mtsamples.csv", "/content/mtsamples.csv"]:
                if Path(p).exists():
                    df = pd.read_csv(p)
                    break
            else:
                raise FileNotFoundError(
                    "MTSamples not found. Upload mtsamples.csv to Colab or check HF availability."
                )

    # Standardize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "transcription" in cl:
            col_map[c] = "text"
        elif cl == "text" and "text" not in col_map.values():
            col_map[c] = "text"
        elif "specialty" in cl or "medical_specialty" in cl:
            col_map[c] = "specialty"
        elif "description" in cl:
            col_map[c] = "description"
        elif "keywords" in cl:
            col_map[c] = "keywords"
        elif "sample_name" in cl or "title" in cl:
            col_map[c] = "title"
    df = df.rename(columns=col_map)

    if "text" not in df.columns:
        for c in df.columns:
            if df[c].dtype == object and df[c].str.len().median() > 200:
                df = df.rename(columns={c: "text"})
                break

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 100].reset_index(drop=True)

    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    print(f"  Loaded {len(df)} notes, {df.get('specialty', pd.Series()).nunique()} specialties")
    return df


def build_mtsamples_queries(df):
    """Build keyword and natural language queries for MTSamples."""
    kw_queries, nl_queries = [], []
    for _, row in df.iterrows():
        parts = []
        if "specialty" in row and pd.notna(row.get("specialty")):
            parts.append(str(row["specialty"]).strip())
        if "keywords" in row and pd.notna(row.get("keywords")):
            kws = str(row["keywords"]).strip()
            kw_list = [k.strip() for k in kws.replace(";", ",").split(",") if k.strip()]
            parts.extend(kw_list[:4])
        elif "title" in row and pd.notna(row.get("title")):
            parts.append(str(row["title"]).strip())
        if not parts:
            parts.append(str(row["text"])[:80])
        kw_queries.append(" ".join(parts))

        if "description" in row and pd.notna(row.get("description")):
            nl_queries.append(str(row["description"]).strip())
        else:
            text = str(row["text"]).strip()
            sents = text.replace("\n", " ").split(". ")
            nl_q = ". ".join(sents[:2])
            if len(nl_q) > 300:
                nl_q = nl_q[:300]
            nl_queries.append(nl_q)
    return kw_queries, nl_queries


# ============================================================
# CELL 5: DATA LOADING — PMC-PATIENTS
# ============================================================
def load_pmc_patients(n=500, seed=42):
    """Load and sample PMC-Patients dataset."""
    from datasets import load_dataset

    print("Loading PMC-Patients...")
    try:
        patients = load_dataset("zhengyun21/PMC-Patients", split="train", token=HF_TOKEN or None)
        patients_df = patients.to_pandas()
    except Exception as e:
        print(f"  load_dataset failed ({e}), falling back to direct JSON load...")
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id="zhengyun21/PMC-Patients",
            filename="PMC-Patients-V2.json",
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        patients_df = pd.read_json(json_path, lines=False)

    print(f"  Total patients: {len(patients_df)}")

    # Find text column
    text_col = None
    for col in ["patient", "text", "patient_text", "summary"]:
        if col in patients_df.columns:
            text_col = col
            break
    if text_col is None:
        for c in patients_df.columns:
            if patients_df[c].dtype == object and patients_df[c].str.len().median() > 100:
                text_col = c
                break
    if text_col is None:
        raise ValueError(f"Cannot find text column. Columns: {list(patients_df.columns)}")

    patients_df = patients_df.rename(columns={text_col: "text"})
    patients_df = patients_df.dropna(subset=["text"])
    patients_df = patients_df[patients_df["text"].str.len() > 50].reset_index(drop=True)

    if len(patients_df) > n:
        patients_df = patients_df.sample(n=n, random_state=seed).reset_index(drop=True)

    print(f"  Sampled {len(patients_df)} patients")
    return patients_df


def build_pmc_queries(df):
    """Build keyword and natural language queries for PMC-Patients."""
    kw_queries, nl_queries = [], []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        words = text.split()

        # Keyword: extract capitalized medical terms
        med_terms = []
        skip_words = {"this", "that", "with", "from", "were", "have", "been",
                      "patient", "year", "years", "history", "presented",
                      "admission", "hospital", "examination", "showed"}
        for w in words:
            clean = w.strip(".,;:()")
            if len(clean) > 3 and (clean[0].isupper() or clean.isupper()):
                if clean.lower() not in skip_words:
                    med_terms.append(clean)
            if len(med_terms) >= 5:
                break
        if not med_terms:
            med_terms = words[:5]
        kw_queries.append(" ".join(med_terms))

        # Natural language: first 1-2 sentences
        sents = text.replace("\n", " ").split(". ")
        nl_q = ". ".join(sents[:2])
        nl_words = nl_q.split()
        if len(nl_words) > 100:
            nl_q = " ".join(nl_words[:100])
        nl_queries.append(nl_q)
    return kw_queries, nl_queries


# ============================================================
# CELL 6: GENERATE SYNTHETIC CLINICAL NOTES
# ============================================================
SPECIALTIES_FOR_SYNTH = [
    "Cardiology", "Orthopedics", "Gastroenterology", "Neurology", "Pulmonology",
    "Endocrinology", "Nephrology", "Oncology", "Urology", "Obstetrics and Gynecology",
    "Dermatology", "Ophthalmology", "ENT / Otolaryngology", "Psychiatry", "Rheumatology",
    "Hematology", "Infectious Disease", "General Surgery", "Emergency Medicine",
    "Family Medicine",
]

SYNTH_PROMPT_TEMPLATE = """<s>[INST] You are a physician writing a clinical note. Write a realistic clinical note for a {specialty} patient encounter. Include:
- Chief Complaint
- History of Present Illness (2-3 paragraphs)
- Past Medical History
- Medications
- Physical Examination findings
- Assessment and Plan

The note should be 200-400 words, use standard medical terminology and abbreviations, and read like a real clinical document. Do not include any patient name or identifiers. Write only the note, no commentary. [/INST]"""


def generate_synthetic_notes(n=500, seed=42):
    """Generate synthetic clinical notes using Mistral-7B-Instruct."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_path = OUTPUT_DIR / "synthetic_notes.csv"
    if cache_path.exists():
        print(f"Loading cached synthetic notes from {cache_path}")
        df = pd.read_csv(cache_path)
        if len(df) >= n:
            return df.head(n)
        print(f"  Cache has {len(df)} notes, need {n}. Regenerating...")

    print(f"Generating {n} synthetic clinical notes with {SYNTH_MODEL}...")
    print("  This may take 15-30 minutes on H100...")

    tokenizer = AutoTokenizer.from_pretrained(SYNTH_MODEL, token=HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        SYNTH_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN or None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = np.random.RandomState(seed)

    # Distribute across specialties
    notes_per_spec = n // len(SPECIALTIES_FOR_SYNTH)
    remainder = n % len(SPECIALTIES_FOR_SYNTH)

    all_notes = []
    for i, specialty in enumerate(tqdm(SPECIALTIES_FOR_SYNTH, desc="Specialties")):
        count = notes_per_spec + (1 if i < remainder else 0)
        for j in range(count):
            prompt = SYNTH_PROMPT_TEMPLATE.format(specialty=specialty)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated note (after [/INST])
            if "[/INST]" in full_text:
                note = full_text.split("[/INST]")[-1].strip()
            else:
                note = full_text[len(prompt):].strip()

            if len(note) > 50:  # skip empty/failed generations
                all_notes.append({
                    "text": note,
                    "specialty": specialty,
                    "synth_id": f"synth_{len(all_notes):04d}",
                })

    df = pd.DataFrame(all_notes)
    print(f"  Generated {len(df)} notes across {df['specialty'].nunique()} specialties")

    # Save cache
    df.to_csv(cache_path, index=False)

    # Free generation model
    del model, tokenizer
    torch.cuda.empty_cache()
    print("  Generation model freed from GPU")

    return df.head(n)


def build_synthetic_queries(df):
    """Build keyword and natural language queries for synthetic notes."""
    kw_queries, nl_queries = [], []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        specialty = str(row.get("specialty", "")).strip()

        # Keyword: specialty + key medical terms
        words = text.split()
        med_terms = [specialty] if specialty else []
        skip_words = {"the", "patient", "with", "and", "was", "for", "this", "that",
                      "chief", "complaint", "history", "present", "illness",
                      "physical", "examination", "assessment", "plan", "past",
                      "medical", "medications", "findings"}
        for w in words:
            clean = w.strip(".,;:()/-")
            if len(clean) > 3 and clean.lower() not in skip_words:
                if clean[0].isupper() or clean.isupper():
                    med_terms.append(clean)
            if len(med_terms) >= 5:
                break
        if not med_terms:
            med_terms = words[:5]
        kw_queries.append(" ".join(med_terms))

        # Natural language: first 1-2 sentences after "Chief Complaint" if present
        lines = text.split("\n")
        nl_q = ""
        for line in lines:
            stripped = line.strip()
            if stripped and "chief complaint" not in stripped.lower() and len(stripped) > 20:
                nl_q = stripped
                break
        if not nl_q:
            sents = text.replace("\n", " ").split(". ")
            nl_q = ". ".join(sents[:2])
        nl_words = nl_q.split()
        if len(nl_words) > 100:
            nl_q = " ".join(nl_words[:100])
        nl_queries.append(nl_q)

    return kw_queries, nl_queries


# ============================================================
# CELL 7: ENCODING FUNCTIONS
# ============================================================
def mean_pooling(hidden_states, attention_mask):
    """Mean pooling over non-padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def eos_pooling(hidden_states, attention_mask):
    """Last non-padding token (EOS) pooling."""
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


def encode_batch_hf(texts, model, tokenizer, config, batch_size=32):
    """Encode texts with a HuggingFace model."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=config.max_length, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            if config.pooling == "cls":
                emb = hidden[:, 0, :]
            elif config.pooling == "eos":
                emb = eos_pooling(hidden, inputs["attention_mask"])
            else:
                emb = mean_pooling(hidden, inputs["attention_mask"])

            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def encode_sentence_transformers(texts, model_id, batch_size=64):
    """Encode using sentence-transformers library."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, trust_remote_code=True)
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return embs


def encode_openai(texts, model_id="text-embedding-3-small", batch_size=100):
    """Encode using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    all_embs = []
    api_log = {"model": model_id, "date": datetime.date.today().isoformat(), "n_texts": len(texts)}

    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI API"):
        batch = texts[i:i+batch_size]
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        response = client.embeddings.create(input=batch, model=model_id)
        batch_embs = [item.embedding for item in response.data]
        all_embs.extend(batch_embs)
        api_log["model_version"] = getattr(response, "model", model_id)

    embs = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)
    return embs, api_log


def encode_with_config(config: ModelConfig, queries: List[str], documents: List[str]):
    """Encode queries and documents according to model config."""
    from transformers import AutoModel, AutoTokenizer

    meta = {"model": config.name, "category": config.category, "pooling": config.pooling}
    print(f"\n{'='*60}")
    print(f"Encoding: {config.name} ({config.category})")
    print(f"  Pooling: {config.pooling} | Prefix: {'yes' if config.query_prefix or config.instruction else 'no'}")
    print(f"{'='*60}")

    # ── OpenAI API ──
    if config.is_api:
        if not OPENAI_API_KEY:
            print(f"  SKIPPING {config.name}: No OPENAI_API_KEY set")
            return None, None, meta
        q_emb, api_log = encode_openai(queries, config.hf_id)
        d_emb, _ = encode_openai(documents, config.hf_id)
        meta["api_log"] = api_log
        return q_emb, d_emb, meta

    # ── Sentence Transformers (Nomic) ──
    if config.use_sentence_transformers:
        prefixed_q = [config.query_prefix + q for q in queries]
        prefixed_d = [config.doc_prefix + d for d in documents]
        q_emb = encode_sentence_transformers(prefixed_q, config.hf_id)
        d_emb = encode_sentence_transformers(prefixed_d, config.hf_id)
        return q_emb, d_emb, meta

    # ── MedCPT Dual Encoder ──
    if config.is_dual_encoder:
        print("  Loading query encoder...")
        q_tokenizer = AutoTokenizer.from_pretrained(config.hf_id_query, token=HF_TOKEN or None)
        q_model = AutoModel.from_pretrained(config.hf_id_query, token=HF_TOKEN or None).to(DEVICE).eval()
        print("  Loading document encoder...")
        d_tokenizer = AutoTokenizer.from_pretrained(config.hf_id_doc, token=HF_TOKEN or None)
        d_model = AutoModel.from_pretrained(config.hf_id_doc, token=HF_TOKEN or None).to(DEVICE).eval()

        q_emb = encode_batch_hf(queries, q_model, q_tokenizer, config)
        d_emb = encode_batch_hf(documents, d_model, d_tokenizer, config)

        del q_model, d_model
        torch.cuda.empty_cache()
        return q_emb, d_emb, meta

    # ── Standard HuggingFace models ──
    print(f"  Loading {config.hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True, token=HF_TOKEN or None)
    load_kwargs = {"trust_remote_code": True, "token": HF_TOKEN or None}
    if config.dtype == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(config.hf_id, **load_kwargs)
    if config.dtype != "fp16":
        model = model.to(DEVICE)
    model.eval()

    # Apply prefixes/instructions
    prefixed_q = []
    for q in queries:
        if config.instruction:
            prefixed_q.append(f"Instruct: {config.instruction}\nQuery: {q}")
        elif config.query_prefix:
            prefixed_q.append(config.query_prefix + q)
        else:
            prefixed_q.append(q)

    prefixed_d = [config.doc_prefix + d if config.doc_prefix else d for d in documents]

    batch_size = 4 if config.dtype == "fp16" else 32
    q_emb = encode_batch_hf(prefixed_q, model, tokenizer, config, batch_size=batch_size)
    d_emb = encode_batch_hf(prefixed_d, model, tokenizer, config, batch_size=batch_size)

    del model
    torch.cuda.empty_cache()
    return q_emb, d_emb, meta


# ============================================================
# CELL 8: RETRIEVAL METRICS
# ============================================================
def compute_retrieval_metrics(q_emb, d_emb, k_values=[1, 5, 10]):
    """Compute retrieval metrics (1-to-1 query-document mapping)."""
    sim_matrix = q_emb @ d_emb.T
    n = sim_matrix.shape[0]
    ranks = []
    for i in range(n):
        scores = sim_matrix[i]
        rank = (scores > scores[i]).sum() + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    metrics = {}
    for k in k_values:
        rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
        metrics[f"MRR@{k}"] = float(rr.mean())
    for k in k_values:
        metrics[f"Recall@{k}"] = float((ranks <= k).mean())
    metrics["P@1"] = float((ranks == 1).mean())

    dcg = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    ideal_dcg = 1.0 / np.log2(2)
    metrics["NDCG@10"] = float(dcg.mean() / ideal_dcg)

    return metrics, ranks, sim_matrix


def bootstrap_ci(ranks, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(ranks)
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_values.append(metric_fn(ranks[idx]))
    boot_values = np.array(boot_values)
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_values, 100 * alpha)), float(np.percentile(boot_values, 100 * (1 - alpha)))


def mrr_at_k(ranks, k=10):
    return np.where(ranks <= k, 1.0 / ranks, 0.0).mean()


# ============================================================
# CELL 9: GEOMETRY METRICS
# ============================================================
def compute_geometry(embeddings, n_pairs=10000, seed=42):
    """Compute embedding space geometry metrics."""
    rng = np.random.RandomState(seed)
    n, d = embeddings.shape

    # Anisotropy
    n_sample = min(n_pairs, n * (n - 1) // 2)
    idx_a = rng.randint(0, n, size=n_sample)
    idx_b = rng.randint(0, n, size=n_sample)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    cos_sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    anisotropy = float(cos_sims.mean())

    # Self-similarity (nearest-neighbor proxy)
    sample_idx = rng.choice(n, size=min(200, n), replace=False)
    sample_emb = embeddings[sample_idx]
    sim_matrix = sample_emb @ sample_emb.T
    np.fill_diagonal(sim_matrix, -1)
    nn_sims = sim_matrix.max(axis=1)
    self_similarity = float(nn_sims.mean())

    # IsoScore (SVD entropy ratio)
    centered = embeddings - embeddings.mean(axis=0)
    if n > 1000:
        sub_idx = rng.choice(n, size=1000, replace=False)
        centered = centered[sub_idx]
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        entropy = -np.sum(S_norm * np.log(S_norm))
        max_entropy = np.log(len(S_norm))
        isoscore = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        eff_rank = float(np.exp(entropy))
    except Exception:
        isoscore, eff_rank = 0.0, 0.0

    return {
        "anisotropy": anisotropy,
        "self_similarity": self_similarity,
        "isoscore": isoscore,
        "effective_rank": eff_rank,
    }


# ============================================================
# CELL 10: MAIN EXPERIMENT LOOP
# ============================================================
def run_experiment():
    """Run 11-model benchmark on 3 datasets."""

    # ── Load all datasets ──
    print("\n" + "=" * 70)
    print("LOADING ALL DATASETS")
    print("=" * 70)

    datasets = {}

    # 1. MTSamples
    try:
        mt_df = load_mtsamples(n=N_SAMPLES, seed=SEED)
        mt_kw, mt_nl = build_mtsamples_queries(mt_df)
        datasets["MTSamples"] = {
            "documents": mt_df["text"].tolist(),
            "queries": {"keyword": mt_kw, "natural_language": mt_nl},
        }
        mt_df.to_csv(OUTPUT_DIR / "mtsamples_sample.csv", index=False)
    except Exception as e:
        print(f"⚠ MTSamples failed: {e}")

    # 2. PMC-Patients
    try:
        pmc_df = load_pmc_patients(n=N_SAMPLES, seed=SEED)
        pmc_kw, pmc_nl = build_pmc_queries(pmc_df)
        datasets["PMC-Patients"] = {
            "documents": pmc_df["text"].tolist(),
            "queries": {"keyword": pmc_kw, "natural_language": pmc_nl},
        }
        pmc_df.to_csv(OUTPUT_DIR / "pmc_patients_sample.csv", index=False)
    except Exception as e:
        print(f"⚠ PMC-Patients failed: {e}")

    # 3. Synthetic (generate first, then free GPU for embeddings)
    try:
        synth_df = generate_synthetic_notes(n=N_SAMPLES, seed=SEED)
        synth_kw, synth_nl = build_synthetic_queries(synth_df)
        datasets["Synthetic"] = {
            "documents": synth_df["text"].tolist(),
            "queries": {"keyword": synth_kw, "natural_language": synth_nl},
        }
    except Exception as e:
        print(f"⚠ Synthetic generation failed: {e}")

    print(f"\nDatasets loaded: {list(datasets.keys())}")
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds['documents'])} documents")

    # ── Results storage ──
    all_results = []

    # ── Run each model × dataset × query format ──
    for model_config in MODELS:
        model_name = model_config.name

        for ds_name, ds_data in datasets.items():
            documents = ds_data["documents"]

            for qf_name, queries in ds_data["queries"].items():
                condition = f"{model_name}__{ds_name}__{qf_name}"
                print(f"\n>>> {condition}")

                try:
                    q_emb, d_emb, meta = encode_with_config(model_config, queries, documents)

                    if q_emb is None:
                        print(f"  Skipped")
                        continue

                    metrics, ranks, sim_matrix = compute_retrieval_metrics(q_emb, d_emb)
                    mrr_lo, mrr_hi = bootstrap_ci(ranks, lambda r: mrr_at_k(r, 10),
                                                   n_bootstrap=N_BOOTSTRAP, seed=SEED)
                    geom = compute_geometry(d_emb)

                    result = {
                        "model": model_name,
                        "category": model_config.category,
                        "dataset": ds_name,
                        "query_format": qf_name,
                        "pooling": model_config.pooling,
                        "has_prefix": bool(model_config.query_prefix or model_config.instruction),
                        **metrics,
                        "MRR@10_lo": mrr_lo,
                        "MRR@10_hi": mrr_hi,
                        **{f"geom_{k}": v for k, v in geom.items()},
                    }
                    all_results.append(result)

                    if SAVE_EMBEDDINGS:
                        safe_name = condition.replace(" ", "_")
                        np.save(OUTPUT_DIR / "embeddings" / f"{safe_name}_q.npy", q_emb)
                        np.save(OUTPUT_DIR / "embeddings" / f"{safe_name}_d.npy", d_emb)

                    with open(OUTPUT_DIR / "metrics" / f"{condition}.json", "w") as f:
                        json.dump(result, f, indent=2)

                    print(f"  MRR@10={metrics['MRR@10']:.3f} [{mrr_lo:.3f}, {mrr_hi:.3f}]  "
                          f"R@5={metrics['Recall@5']:.3f}  P@1={metrics['P@1']:.3f}  "
                          f"Aniso={geom['anisotropy']:.3f}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {len(results_df)} conditions across {len(datasets)} datasets")
    print(f"{'='*70}")
    return results_df


# ============================================================
# CELL 11: ANALYSIS & FIGURES
# ============================================================
def generate_analysis(results_df):
    """Generate figures and summary tables from 3-dataset results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    cat_colors = {
        "Domain Encoder": "#c0392b", "Biomedical Retriever": "#e67e22",
        "General Embedding": "#27ae60", "General API": "#8e44ad",
        "General LLM": "#2980b9",
    }

    figdir = OUTPUT_DIR / "figures"
    primary = results_df[~results_df["model"].str.contains("ablation|nopfx", case=False)].copy()
    ds_list = sorted(primary["dataset"].unique())
    n_ds = len(ds_list)

    # ── FIGURE 1: MRR bar chart per dataset (keyword queries) ──
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=True)
    if n_ds == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, ds_list):
        sub = primary[(primary["dataset"] == ds_name) & (primary["query_format"] == "keyword")]
        sub = sub.sort_values("MRR@10", ascending=True)
        if sub.empty:
            continue
        colors = [cat_colors.get(c, "#999") for c in sub["category"]]
        bars = ax.barh(
            range(len(sub)), sub["MRR@10"].values,
            color=colors, edgecolor="black", linewidth=0.6, height=0.65,
            xerr=[sub["MRR@10"].values - sub["MRR@10_lo"].values,
                  sub["MRR@10_hi"].values - sub["MRR@10"].values],
            error_kw={"linewidth": 0.8, "capsize": 2},
        )
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["model"].values, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("MRR@10")
        ax.set_title(ds_name, fontweight="bold", fontsize=11)
        for bar, p1 in zip(bars, sub["P@1"].values):
            ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                    f"P@1={p1:.0%}", va="center", fontsize=5.5, color="#555")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=cat_colors.get(c, "#999"), ec="black", lw=0.6)
               for c in cat_colors if c in primary["category"].values]
    labels = [c for c in cat_colors if c in primary["category"].values]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figdir / "fig1_mrr_all_datasets.png")
    fig.savefig(figdir / "fig1_mrr_all_datasets.pdf")
    plt.close()
    print("  Fig 1: MRR bar charts (3 datasets) saved")

    # ── FIGURE 2: Cross-dataset rank correlation matrix ──
    kw = primary[primary["query_format"] == "keyword"]
    if n_ds >= 2:
        pivot = kw.pivot_table(index="model", columns="dataset", values="MRR@10")
        fig, axes = plt.subplots(1, max(1, n_ds * (n_ds - 1) // 2), figsize=(5.5 * max(1, n_ds * (n_ds - 1) // 2), 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        pair_idx = 0
        corr_results = {}
        for i in range(len(ds_list)):
            for j in range(i + 1, len(ds_list)):
                ds_a, ds_b = ds_list[i], ds_list[j]
                if ds_a not in pivot.columns or ds_b not in pivot.columns:
                    continue
                valid = pivot[[ds_a, ds_b]].dropna()
                if len(valid) < 4:
                    continue

                rho, p_val = spearmanr(valid[ds_a], valid[ds_b])
                corr_results[f"{ds_a} vs {ds_b}"] = {"rho": rho, "p": p_val}

                ax = axes[pair_idx] if pair_idx < len(axes) else axes[-1]
                for _, row in valid.iterrows():
                    model_name = row.name
                    cat = primary[primary["model"] == model_name]["category"].iloc[0]
                    ax.scatter(row[ds_a], row[ds_b],
                              c=cat_colors.get(cat, "#999"), s=120,
                              edgecolors="black", linewidth=0.8, zorder=5)
                    ax.annotate(model_name, (row[ds_a], row[ds_b]),
                               textcoords="offset points", xytext=(5, 4), fontsize=6.5)

                lims = [0, max(valid[ds_a].max(), valid[ds_b].max()) + 0.05]
                ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
                ax.set_xlabel(f"MRR@10 — {ds_a}")
                ax.set_ylabel(f"MRR@10 — {ds_b}")
                ax.set_title(f"ρ = {rho:.3f}, p = {p_val:.4f}", fontweight="bold", fontsize=10)
                pair_idx += 1

        plt.tight_layout()
        fig.savefig(figdir / "fig2_cross_dataset_correlation.png")
        fig.savefig(figdir / "fig2_cross_dataset_correlation.pdf")
        plt.close()

        print("  Fig 2: Cross-dataset correlations saved")
        for pair, vals in corr_results.items():
            print(f"    {pair}: ρ={vals['rho']:.3f}, p={vals['p']:.4f}")

        with open(OUTPUT_DIR / "cross_dataset_correlations.json", "w") as f:
            json.dump(corr_results, f, indent=2)

    # ── FIGURE 3: Geometry scatter (anisotropy vs MRR, all datasets) ──
    if not kw.empty:
        ds_markers = {"MTSamples": "o", "PMC-Patients": "s", "Synthetic": "^"}
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for cat in cat_colors:
            for ds in ds_list:
                mask = (kw["category"] == cat) & (kw["dataset"] == ds)
                sub = kw[mask]
                if sub.empty:
                    continue
                ax.scatter(sub["geom_anisotropy"], sub["MRR@10"],
                          c=cat_colors[cat], s=100, marker=ds_markers.get(ds, "o"),
                          edgecolors="black", linewidth=0.6, zorder=5,
                          label=f"{cat} ({ds})" if ds == ds_list[0] else None)
                for _, row in sub.iterrows():
                    ax.annotate(row["model"], (row["geom_anisotropy"], row["MRR@10"]),
                               textcoords="offset points", xytext=(5, 3), fontsize=5.5, color="#333")

        # Legend for categories
        cat_handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6) for c in cat_colors.values()
                       if any(kw["category"] == k for k in cat_colors)]
        cat_labels = [k for k in cat_colors if any(kw["category"] == k)]
        # Legend for datasets
        from matplotlib.lines import Line2D
        ds_handles = [Line2D([0], [0], marker=ds_markers.get(d, "o"), color="gray",
                            markersize=8, linestyle="None") for d in ds_list]

        l1 = ax.legend(cat_handles, cat_labels, loc="upper right", fontsize=7, title="Category")
        l2 = ax.legend(ds_handles, ds_list, loc="upper left", fontsize=7, title="Dataset")
        ax.add_artist(l1)

        ax.set_xlabel("Anisotropy (higher → more collapsed)")
        ax.set_ylabel("MRR@10 (keyword queries)")
        ax.set_title("Embedding Geometry vs Retrieval Quality\nAcross Three Clinical Text Genres", fontweight="bold")
        plt.tight_layout()
        fig.savefig(figdir / "fig3_geometry_scatter.png")
        fig.savefig(figdir / "fig3_geometry_scatter.pdf")
        plt.close()
        print("  Fig 3: Geometry scatter saved")

    # ── FIGURE 4: Query sensitivity per dataset ──
    for ds in ds_list:
        ds_data = primary[primary["dataset"] == ds]
        models = ds_data["model"].unique()
        delta_rows = []
        for m in models:
            kw_row = ds_data[(ds_data["model"] == m) & (ds_data["query_format"] == "keyword")]
            nl_row = ds_data[(ds_data["model"] == m) & (ds_data["query_format"] == "natural_language")]
            if kw_row.empty or nl_row.empty:
                continue
            delta_rows.append({
                "model": m, "category": kw_row.iloc[0]["category"],
                "dMRR": nl_row.iloc[0]["MRR@10"] - kw_row.iloc[0]["MRR@10"],
            })
        if delta_rows:
            df_delta = pd.DataFrame(delta_rows).sort_values("dMRR")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = [cat_colors.get(c, "#999") for c in df_delta["category"]]
            bars = ax.barh(range(len(df_delta)), df_delta["dMRR"].values,
                          color=colors, edgecolor="black", linewidth=0.6, height=0.6)
            ax.set_yticks(range(len(df_delta)))
            ax.set_yticklabels(df_delta["model"].values, fontsize=8)
            ax.axvline(x=0, color="black", linewidth=1)
            ax.set_xlabel("ΔMRR (natural language − keyword)")
            ax.set_title(f"Query Sensitivity — {ds}", fontweight="bold")
            for bar, val in zip(bars, df_delta["dMRR"].values):
                x = bar.get_width()
                ax.text(x + (0.003 if x >= 0 else -0.003), bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha="left" if x >= 0 else "right",
                        fontsize=7, fontweight="bold")
            plt.tight_layout()
            safe = ds.replace(" ", "_").replace("-", "_")
            fig.savefig(figdir / f"fig4_sensitivity_{safe}.png")
            fig.savefig(figdir / f"fig4_sensitivity_{safe}.pdf")
            plt.close()
    print("  Fig 4: Query sensitivity saved")

    # ── ABLATION RESULTS ──
    ablation = results_df[results_df["model"].str.contains("ablation|nopfx", case=False)]
    if not ablation.empty:
        print("\n  ABLATION RESULTS:")
        for _, row in ablation.iterrows():
            print(f"    {row['model']} | {row['dataset']} | {row['query_format']} | MRR@10={row['MRR@10']:.3f}")

    # ── SUMMARY TABLE ──
    print("\n" + "=" * 70)
    print("SUMMARY: PRIMARY MODELS, KEYWORD QUERIES")
    print("=" * 70)
    summary = primary[primary["query_format"] == "keyword"][
        ["model", "category", "dataset", "MRR@10", "MRR@10_lo", "MRR@10_hi",
         "Recall@5", "P@1", "geom_anisotropy"]
    ].sort_values(["dataset", "MRR@10"], ascending=[True, False])
    print(summary.to_string(index=False, float_format="%.3f"))

    # ── Spearman: anisotropy vs MRR per dataset ──
    for ds in ds_list:
        kw_ds = primary[(primary["dataset"] == ds) & (primary["query_format"] == "keyword")]
        if len(kw_ds) >= 4:
            rho, p = spearmanr(kw_ds["geom_anisotropy"], kw_ds["MRR@10"])
            print(f"\n  Spearman (anisotropy vs MRR@10, {ds}): ρ={rho:.3f}, p={p:.4f}")

    return primary


# ============================================================
# CELL 12: ENVIRONMENT LOG
# ============================================================
def save_env_log():
    """Save environment info for reproducibility."""
    import platform, sys
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_bootstrap": N_BOOTSTRAP,
        "synth_model": SYNTH_MODEL,
    }
    try:
        import transformers, sentence_transformers
        log["transformers"] = transformers.__version__
        log["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
    with open(OUTPUT_DIR / "env_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Environment log saved")


# ============================================================
# CELL 13: RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    save_env_log()
    results_df = run_experiment()
    primary = generate_analysis(results_df)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total conditions: {len(results_df)}")
    print(f"\nFiles:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1e6:
                print(f"  {f.relative_to(OUTPUT_DIR)}  ({size/1e6:.1f} MB)")
            else:
                print(f"  {f.relative_to(OUTPUT_DIR)}  ({size/1e3:.1f} KB)")
