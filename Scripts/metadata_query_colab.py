"""
Metadata-Only Query Validation — Google Colab (H100)
Upload your corpus files + all_results.csv to /content/sample_data/ before running.
"""

# %%
!pip install -q sentence-transformers transformers rank-bm25 openai tiktoken tqdm

# %%
import os, json, re, time, gc, string, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASTE YOUR KEY HERE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPENAI_API_KEY = "sk-paste-your-key-here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALL FILES GO IN /content/sample_data/
# Upload there before running:
#   - Your corpus JSONL or CSV files
#   - all_results.csv (your 294-condition results)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_DIR = Path("/content/sample_data")
RESULTS_DIR = Path("/content/sample_data/results")
RESULTS_DIR.mkdir(exist_ok=True)

N_DOCS_PER_CORPUS = 100
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")

# Show what's in sample_data
print(f"\nFiles in {DATA_DIR}:")
for f in sorted(DATA_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        print(f"  {f.name}  ({size/1024:.0f} KB)")

# %% [markdown]
# # 1. Load Corpora

# %%
def load_file(path):
    """Load JSONL or CSV into list of dicts."""
    if path.suffix == ".jsonl":
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        # Find text column
        text_col = next((c for c in df.columns
                         if c.lower() in ("text","transcription","document","note","content")),
                        df.columns[-1])
        df = df.dropna(subset=[text_col])
        spec_col = next((c for c in df.columns
                         if "special" in c.lower()), None)
        type_col = next((c for c in df.columns
                         if "type" in c.lower() or "sample" in c.lower()), None)
        corpus_name = path.stem.split("_")[0].title()
        docs = []
        for i, row in df.iterrows():
            txt = str(row[text_col]).strip()
            if len(txt.split()) >= 30:
                docs.append({
                    "doc_id": f"{corpus_name}_{i}",
                    "corpus": corpus_name,
                    "text": txt,
                    "specialty": str(row[spec_col]).strip() if spec_col else "Unknown",
                    "note_type": str(row[type_col]).strip() if type_col else "Clinical Note",
                })
        return docs
    else:
        raise ValueError(f"Unsupported: {path.suffix}")

# Auto-detect corpus files in sample_data
corpus_keywords = {
    "MTSamples": ["mtsample", "mt_sample", "mtsamples"],
    "PMC-Patients": ["pmc", "patient"],
    "Synthetic": ["synth", "generated"],
}

corpus_files = {}
for f in DATA_DIR.iterdir():
    if f.suffix in (".jsonl", ".csv") and "result" not in f.name.lower():
        name_lower = f.name.lower()
        for corpus_name, keywords in corpus_keywords.items():
            if any(k in name_lower for k in keywords):
                corpus_files[corpus_name] = f
                break

print("Detected corpus files:")
for name, path in corpus_files.items():
    print(f"  {name} → {path.name}")

if not corpus_files:
    print("\n⚠️  No corpus files detected!")
    print("Upload JSONL/CSV files to /content/sample_data/ with names containing")
    print("'mtsample', 'pmc', or 'synth'.")
    raise FileNotFoundError("No corpus files found in /content/sample_data/")

# Load and sample
corpora = {}
for name, path in corpus_files.items():
    all_docs = load_file(path)
    rng = np.random.RandomState(SEED)
    n = min(N_DOCS_PER_CORPUS, len(all_docs))
    idx = rng.choice(len(all_docs), size=n, replace=False)
    corpora[name] = [all_docs[i] for i in idx]
    print(f"{name}: {len(all_docs)} total → sampled {n}")

# Find original results CSV
orig_csv = None
for f in DATA_DIR.iterdir():
    if f.suffix == ".csv" and "result" in f.name.lower():
        orig_csv = f
        break
print(f"\nOriginal results: {orig_csv.name if orig_csv else 'NOT FOUND'}")

# %% [markdown]
# # 2. Extract Metadata (GPT-4)

# %%
from openai import OpenAI
client = OpenAI()

META_PROMPT = """Extract metadata from this clinical document as JSON:
{"specialty":"...","note_type":"...","primary_diagnosis":"...","secondary_diagnoses":["..."],"patient_demographics":"..."}
Return ONLY valid JSON.

Document:
{text}"""

def get_metadata(text):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o", temperature=0, max_tokens=300,
                messages=[{"role":"user","content":META_PROMPT.format(text=text[:3000])}])
            raw = r.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*','',raw)
            raw = re.sub(r'\s*```$','',raw)
            return json.loads(raw)
        except:
            time.sleep(2**attempt)
    return {"specialty":"Unknown","note_type":"Clinical Note",
            "primary_diagnosis":"Unknown","secondary_diagnoses":[],"patient_demographics":"adult"}

all_docs = []
for cn, docs in corpora.items():
    print(f"\n{cn}:")
    for i, doc in enumerate(tqdm(docs, desc=cn)):
        meta = get_metadata(doc["text"])
        all_docs.append({"doc_id":doc.get("doc_id",f"{cn}_{i}"),"corpus":cn,
                         "text":doc["text"],"metadata":meta})
        time.sleep(0.3)

with open(RESULTS_DIR/"metadata.jsonl","w") as f:
    for d in all_docs:
        f.write(json.dumps({"doc_id":d["doc_id"],"corpus":d["corpus"],"metadata":d["metadata"]})+"\n")
print(f"\nExtracted metadata for {len(all_docs)} documents")

# %% [markdown]
# # 3. Generate Metadata-Only Queries
# GPT-4 sees ONLY metadata — never the document text.

# %%
NL_TPL = """Based ONLY on this metadata, write a natural language clinical question (1-2 sentences):
Specialty: {sp} | Note type: {nt} | Diagnosis: {dx} | Other: {sec} | Patient: {dem}
Query:"""

KW_TPL = """Based ONLY on this metadata, output 3-6 clinical search keywords (space-separated):
Specialty: {sp} | Note type: {nt} | Diagnosis: {dx} | Other: {sec} | Patient: {dem}
Keywords:"""

def gen_q(meta, tpl):
    sec = ", ".join(meta.get("secondary_diagnoses",[]) or []) or "none"
    prompt = tpl.format(sp=meta.get("specialty","Unknown"), nt=meta.get("note_type","Note"),
                        dx=meta.get("primary_diagnosis","Unknown"), sec=sec,
                        dem=meta.get("patient_demographics","adult"))
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o", temperature=0.3, max_tokens=150,
                messages=[{"role":"user","content":prompt}])
            return r.choices[0].message.content.strip()
        except:
            time.sleep(2**attempt)
    return meta.get("primary_diagnosis","clinical query")

queries = []
for doc in tqdm(all_docs, desc="Queries"):
    m = doc["metadata"]
    queries.append({"doc_id":doc["doc_id"],"corpus":doc["corpus"],
                    "nl_query":gen_q(m,NL_TPL),"kw_query":gen_q(m,KW_TPL)})
    time.sleep(0.3)

with open(RESULTS_DIR/"metadata_queries.jsonl","w") as f:
    for q in queries:
        f.write(json.dumps(q)+"\n")
print(f"Generated {len(queries)} query pairs")

# %% [markdown]
# # 4. Retrieval Evaluation

# %%
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

STOP = {"a","an","the","is","are","was","were","be","been","have","has","had","do",
        "does","did","will","would","shall","should","may","might","must","can","could",
        "i","me","my","we","our","you","your","he","him","his","she","her","it","its",
        "they","them","their","this","that","these","those","in","on","at","to","for",
        "of","with","by","from","as","and","but","or","nor","not","so","very","just","if"}

def tok_bm25(t):
    return [w for w in t.lower().translate(str.maketrans("","",string.punctuation)).split()
            if w not in STOP and len(w)>1]

def clear():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def mrr(ranks,k=10): return np.mean([1/r if r<=k else 0 for r in ranks])
def hr(ranks,k=10):  return np.mean([1 if r<=k else 0 for r in ranks])

def boot_ci(ranks,fn,k=10,n=1000):
    rng=np.random.RandomState(SEED)
    v=[fn(rng.choice(ranks,len(ranks),replace=True).tolist(),k) for _ in range(n)]
    return np.percentile(v,2.5),np.percentile(v,97.5)

def rankings_from_sims(q_embs, d_embs):
    sims = q_embs @ d_embs.T
    return [int(np.where(np.argsort(-sims[i])==i)[0][0])+1 for i in range(len(q_embs))]

def enc_st(hub, texts, prefix="", trust=False):
    kw = {"trust_remote_code":True} if trust else {}
    m = SentenceTransformer(hub, device=DEVICE, **kw)
    e = m.encode([prefix+t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    del m; clear()
    return e

def enc_openai(texts, model="text-embedding-3-small"):
    out=[]
    for i in range(0,len(texts),100):
        r=client.embeddings.create(model=model,input=texts[i:i+100])
        out.extend([e.embedding for e in r.data]); time.sleep(0.5)
    return np.array(out)

def enc_medcpt(qtexts, dtexts):
    from transformers import AutoTokenizer, AutoModel
    res=[]
    for hub,texts in [("ncbi/MedCPT-Query-Encoder",qtexts),("ncbi/MedCPT-Article-Encoder",dtexts)]:
        tok=AutoTokenizer.from_pretrained(hub)
        mdl=AutoModel.from_pretrained(hub).to(DEVICE).eval()
        embs=[]
        for i in range(0,len(texts),32):
            enc=tok(texts[i:i+32],padding=True,truncation=True,max_length=512,return_tensors="pt").to(DEVICE)
            with torch.no_grad(): e=mdl(**enc).last_hidden_state[:,0]
            embs.append(torch.nn.functional.normalize(e,p=2,dim=1).cpu().numpy())
        del mdl,tok; clear()
        res.append(np.vstack(embs))
    return res[0],res[1]

def enc_llm(hub,texts,prefix="",pooling="mean"):
    from transformers import AutoTokenizer, AutoModel
    tok=AutoTokenizer.from_pretrained(hub)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    mdl=AutoModel.from_pretrained(hub,torch_dtype=torch.float16).to(DEVICE).eval()
    embs=[]
    for i in range(0,len(texts),4):
        batch=[prefix+t for t in texts[i:i+4]]
        enc=tok(batch,padding=True,truncation=True,max_length=4096,return_tensors="pt").to(DEVICE)
        with torch.no_grad(): out=mdl(**enc)
        if pooling=="last_token":
            sl=enc["attention_mask"].sum(dim=1)-1
            e=out.last_hidden_state[torch.arange(len(sl)),sl]
        else:
            mask=enc["attention_mask"].unsqueeze(-1)
            e=(out.last_hidden_state*mask).sum(1)/mask.sum(1)
        embs.append(torch.nn.functional.normalize(e,p=2,dim=1).cpu().numpy())
    del mdl,tok; clear()
    return np.vstack(embs)

# Model configs
MODELS = {
    "BM25":             {"tp":"bm25"},
    "BioBERT":          {"tp":"st","hub":"dmis-lab/biobert-base-cased-v1.2","qp":"","dp":""},
    "ClinicalBERT":     {"tp":"st","hub":"emilyalsentzer/Bio_ClinicalBERT","qp":"","dp":""},
    "BioLORD-2023":     {"tp":"st","hub":"FremyCompany/BioLORD-2023","qp":"","dp":""},
    "BGE-base":         {"tp":"st","hub":"BAAI/bge-base-en-v1.5",
                         "qp":"Represent this sentence for searching relevant passages: ","dp":""},
    "GTE-base":         {"tp":"st","hub":"Alibaba-NLP/gte-base-en-v1.5","qp":"","dp":""},
    "Nomic-embed-text": {"tp":"st","hub":"nomic-ai/nomic-embed-text-v1.5",
                         "qp":"search_query: ","dp":"search_document: ","trust":True},
    "Nomic-nopfx":      {"tp":"st","hub":"nomic-ai/nomic-embed-text-v1.5","qp":"","dp":"","trust":True},
    "OpenAI-emb3-small":{"tp":"openai","model":"text-embedding-3-small"},
    "MedCPT":           {"tp":"medcpt"},
    "E5-Mistral-7B":    {"tp":"llm","hub":"intfloat/e5-mistral-7b-instruct",
                         "qp":"Instruct: Retrieve a clinical document relevant to this query\nQuery: ",
                         "dp":"","pool":"last_token"},
    "E5-Mistral-7B-meanpool":{"tp":"llm","hub":"intfloat/e5-mistral-7b-instruct",
                         "qp":"Instruct: Retrieve a clinical document relevant to this query\nQuery: ",
                         "dp":"","pool":"mean"},
    "Phi-3-mini":       {"tp":"llm","hub":"microsoft/Phi-3-mini-128k-instruct","qp":"","dp":"","pool":"mean"},
}

# %%
# ── Run ──
print("="*60)
print("EVALUATION")
print("="*60)

results = []
for mname, cfg in MODELS.items():
    print(f"\n{'─'*40}\n{mname}\n{'─'*40}")
    for cn, docs in corpora.items():
        dtexts = [d["text"] for d in docs]
        cq = [q for q in queries if q["corpus"]==cn]

        # Precompute doc embeddings
        d_embs = None
        if cfg["tp"]=="st":
            d_embs = enc_st(cfg["hub"], dtexts, cfg["dp"], cfg.get("trust",False))
        elif cfg["tp"]=="openai":
            d_embs = enc_openai(dtexts, cfg["model"])
        elif cfg["tp"]=="llm":
            d_embs = enc_llm(cfg["hub"], dtexts, cfg["dp"], cfg.get("pool","mean"))

        for qt in ["kw","nl"]:
            qkey = "kw_query" if qt=="kw" else "nl_query"
            qtexts = [q[qkey] for q in cq]
            label = "KW" if qt=="kw" else "NL"
            print(f"  {cn}/{label}...", end=" ", flush=True)

            if cfg["tp"]=="bm25":
                tokenized=[tok_bm25(d) for d in dtexts]
                bm25=BM25Okapi(tokenized)
                ranks=[int(np.where(np.argsort(-bm25.get_scores(tok_bm25(q)))==i)[0][0])+1
                       for i,q in enumerate(qtexts)]
            elif cfg["tp"]=="openai":
                q_embs=enc_openai(qtexts, cfg["model"])
                ranks=rankings_from_sims(q_embs, d_embs)
            elif cfg["tp"]=="medcpt":
                q_embs, d_embs_m = enc_medcpt(qtexts, dtexts)
                ranks=rankings_from_sims(q_embs, d_embs_m)
            elif cfg["tp"]=="llm":
                q_embs=enc_llm(cfg["hub"], qtexts, cfg["qp"], cfg.get("pool","mean"))
                ranks=rankings_from_sims(q_embs, d_embs)
            else:  # st
                q_embs=enc_st(cfg["hub"], qtexts, cfg["qp"], cfg.get("trust",False))
                ranks=rankings_from_sims(q_embs, d_embs)

            m_val=mrr(ranks); ci=boot_ci(ranks,mrr)
            print(f"MRR@10={m_val:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
            results.append({"model":mname,"corpus":cn,"query_type":qt,
                            "MRR@10":m_val,"HitRate@10":hr(ranks),
                            "CI_lo":ci[0],"CI_hi":ci[1],"median_rank":float(np.median(ranks))})
        clear()

rdf = pd.DataFrame(results)
rdf.to_csv(RESULTS_DIR/"metadata_query_results.csv", index=False)
print(f"\nSaved {len(rdf)} rows")

# %% [markdown]
# # 5. Compare With Original Results

# %%
print("="*60)
print("COMPARISON")
print("="*60)

if orig_csv and orig_csv.exists():
    orig = pd.read_csv(orig_csv)
    # Filter full-document
    if "chunk_strategy" in orig.columns:
        orig = orig[orig["chunk_strategy"]=="full"].copy()
    # Normalize columns
    rename = {}
    for c in orig.columns:
        cl = c.lower()
        if "mrr" in cl and "10" in cl: rename[c]="MRR@10_orig"
        if cl in ("dataset",): rename[c]="corpus"
        if "query" in cl and "format" in cl: rename[c]="qf_orig"
    orig = orig.rename(columns=rename)
    if "qf_orig" in orig.columns:
        orig["query_type"] = orig["qf_orig"].apply(lambda x: "kw" if "keyword" in str(x).lower() else "nl")

    comp = rdf.merge(orig[["model","corpus","query_type","MRR@10_orig"]].drop_duplicates(),
                     on=["model","corpus","query_type"], how="left")

    print("\n── Per-Condition ──")
    print(comp[["model","corpus","query_type","MRR@10","MRR@10_orig"]].to_string(index=False))

    print("\n── Rank Correlations ──")
    for cn in corpora:
        for qt in ["kw","nl"]:
            s = comp[(comp["corpus"]==cn)&(comp["query_type"]==qt)].dropna(subset=["MRR@10_orig"])
            if len(s)>=5:
                tau,pt=stats.kendalltau(s["MRR@10"],s["MRR@10_orig"])
                rho,pr=stats.spearmanr(s["MRR@10"],s["MRR@10_orig"])
                print(f"  {cn}/{'KW' if qt=='kw' else 'NL'}: τ={tau:.3f}(p={pt:.3f}) ρ={rho:.3f}(p={pr:.3f})")

    print("\n── BM25 Drop ──")
    bm=comp[comp["model"]=="BM25"]
    for _,r in bm.iterrows():
        if pd.notna(r.get("MRR@10_orig")):
            print(f"  {r['corpus']}/{'KW' if r['query_type']=='kw' else 'NL'}: "
                  f"{r['MRR@10_orig']:.3f}→{r['MRR@10']:.3f} (Δ={r['MRR@10']-r['MRR@10_orig']:+.3f})")

    merged=comp.dropna(subset=["MRR@10_orig"])
    if len(merged)>0:
        print(f"\n  Overall: metadata={merged['MRR@10'].mean():.3f}, original={merged['MRR@10_orig'].mean():.3f}")
else:
    print("No original results CSV found for comparison.")

# %%
print(f"\n✅ Done. Results in {RESULTS_DIR}/")
try:
    from google.colab import files as gf
    gf.download(str(RESULTS_DIR/"metadata_query_results.csv"))
except: pass
