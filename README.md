# ♻️ E-Waste Policy Assistant — Maryland 2026

A RAG-powered chatbot that answers questions about Maryland e-waste legislation, EPA Universal Waste rules, Baltimore DPW protocols, and NFPA 855 safety standards — with citations to specific bill sections.

**Live app:** [Streamlit Cloud →](https://share.streamlit.io) *(connect `sriram369/ewaste-policy-bot`)*

---

## What It Does

| Tab | Feature |
|---|---|
| **Policy Chat** | Ask questions about HB 992, SB 655, SB 553, EPA, NFPA 855 — always cites the specific section |
| **Metal Calculator** | Estimate scrap metal recovery value (Gold, Palladium, Copper) at real Feb 2026 spot prices |
| **Document Upload** | Upload a new PDF (e.g. local ordinance) and make it immediately searchable |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│                                                                 │
│  Tab 1: Policy Chat    Tab 2: Metal Calc    Tab 3: Upload PDF   │
└────────────┬────────────────────────────────────────────────────┘
             │ user message
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    respond() router  (brain.py)                 │
│                                                                 │
│   _is_policy_question()  ──────────────────────────────────┐   │
│        │ yes                                                │   │
│        ▼                                              no (casual)│
│   query_rag()                                              │   │
│        │                                                   ▼   │
│   FAISS retriever (k=8)               CONVERSATIONAL_PROMPT    │
│        │                                   │                   │
│        ▼                                   ▼                   │
│   top-8 chunks            ChatGroq (llama-3.3-70b-versatile)   │
│        │                                   │                   │
│        ▼                                   │                   │
│   CITATION_PROMPT                          │                   │
│        │                                   │                   │
│        ▼                                   │                   │
│   ChatGroq LLM ──────────────────────────┘                    │
│   (standard: llama-3.3-70b-versatile)                          │
│   (complex:  openai/gpt-oss-120b)                              │
└─────────────────────────────────────────────────────────────────┘
             │ cited answer
             ▼
        User sees response
```

### RAG Pipeline Detail

```
knowledge_base/
├── MD HB 992 SB 655/          ─┐
├── MD SB 553 HB 833/            │
├── EPA Feb 2026/                ├─► PyPDFDirectoryLoader (recursive)
├── Baltimore DPW/               │       │
└── NFPA 855/                   ─┘       ▼
                                    10 PDFs → 487 chunks
                                         │
                                         ▼
                              RecursiveCharacterTextSplitter
                              (chunk_size=1000, overlap=200)
                                         │
                                         ▼
                              HuggingFace all-MiniLM-L6-v2
                              (384-dim embeddings, CPU, no key)
                                         │
                                         ▼
                                   FAISS index
                                  (faiss_db/ on disk)
                                         │
                              ┌──────────┴──────────┐
                         Fast path                First run
                      (loads from disk)       (builds + saves)
```

### Embedding & LLM Stack

```
Component          | Provider             | Model
───────────────────┼──────────────────────┼─────────────────────────────
Embeddings         | HuggingFace (CPU)    | all-MiniLM-L6-v2 (384-dim)
LLM — Standard     | Groq API             | llama-3.3-70b-versatile
LLM — Complex      | Groq API             | openai/gpt-oss-120b (120B)
Vector store       | FAISS                | faiss-cpu (pre-built index)
```

---

## Knowledge Base

The chatbot answers from **10 PDFs across 5 folders**:

```
knowledge_base/
├── MD HB 992 SB 655 (2026) — Oct 1st Cliff/
│   ├── Maryland-2026-HB992-Introduced.pdf
│   └── SENATE BILL 655.pdf
├── MD SB 553 HB 833 (2026) — Lithium-Ion Battery Safety/
│   ├── hb0833f.pdf
│   └── sb0553f.pdf
├── EPA Feb 2026 — Universal Waste NPRM/
│   ├── universal-waste_olem_oct-2024_final.pdf
│   └── (EPA web article PDF)
├── Baltimore DPW Protocols/
│   ├── disposalregs.pdf
│   └── lwbbtask3reportfinal.pdf
└── NFPA 855 (2026)/
    ├── (Telgian summary PDF)
    └── (MeyerFire summary PDF)
```

> PDFs are gitignored and must be added manually. The pre-built FAISS index (`faiss_db/`) is committed — so the app loads instantly without re-ingesting on startup.

---

## File Structure

```
Policy BOT/
├── app.py                    # Streamlit UI — 3 tabs wired together
├── brain.py                  # RAG engine — FAISS, Groq, routing, citations
├── metal_calculator.py       # Scrap metal recovery (Feb 2026 spot prices)
├── evaluate.py               # RAGAS evaluation script
├── requirements.txt          # All Python dependencies
├── checker.md                # Full build log — step-by-step history
├── faiss_db/                 # Pre-built FAISS index (487 chunks, 384-dim)
│   ├── index.faiss
│   └── index.pkl
├── knowledge_base/           # Source PDFs (5 subfolders, 10 PDFs)
│   └── uploads/              # PDFs uploaded via Tab 3 (gitignored)
└── .streamlit/
    ├── config.toml           # Dark GovTech theme
    └── secrets.toml          # GROQ_API_KEY — local only, gitignored
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/sriram369/ewaste-policy-bot.git
cd ewaste-policy-bot
```

### 2. Install Python dependencies

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com).

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> This file is gitignored — it will never be committed.

### 4. Add PDFs to the knowledge base

Copy your 10 PDFs into the correct subfolders under `knowledge_base/` (see structure above).

The pre-built FAISS index (`faiss_db/`) is already committed, so you can skip this step if you want to use the pre-indexed content. If you add new PDFs, delete `faiss_db/` so the app rebuilds the index on next startup.

### 5. Run the app

```bash
python3 -W ignore -m streamlit run app.py
```

App opens at **http://localhost:8501**.

> The `-W ignore` flag suppresses deprecation warnings from LangChain internals — the app runs fine without it, just noisier logs.

---

## Streamlit Cloud Deployment

### 1. Fork or push the repo

The repo must be on GitHub. The FAISS index (`faiss_db/`) must be committed so the cloud app doesn't need to rebuild it.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select `sriram369/ewaste-policy-bot` → `main` branch → `app.py`
4. Click **Deploy**

### 3. Set your Groq API key in Streamlit secrets

1. After deploying, click **Manage app** (bottom-right corner)
2. Go to **Settings → Secrets**
3. Paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

4. Click **Save** — the app will restart automatically

### 4. Done

The app loads the pre-built FAISS index from the repo — no PDF ingestion needed on the cloud. Cold start takes ~30 seconds for the HuggingFace embedding model to load.

---

## How the Routing Works

Every user message goes through `_is_policy_question()` before touching the vector store:

```
User message
     │
     ▼
Is it > 3 words AND contains a policy keyword?
(bill, hb, sb, require, compliance, battery, ewaste, recycl, disposal, ...)
     │
  ┌──┴──┐
 YES    NO
  │      │
  ▼      ▼
RAG   Direct LLM (conversational)
(citations)  (friendly response)
```

This means greetings and general questions get natural responses instead of "not in knowledge base."

---

## Metal Calculator — Feb 2026 Rates

| Metal | Spot Price | Yield / Ton Mixed E-Waste | Value / Ton |
|---|---|---|---|
| Gold | $5,006 / troy oz | 0.50 troy oz | $2,503.00 |
| Palladium | $1,752 / troy oz | 0.05 troy oz | $87.60 |
| Copper | $4.50 / lb scrap | 70 lbs | $315.00 |
| **Total** | | | **$2,905.60** |

---

## Evaluation (RAGAS)

Evaluated against 12 ground-truth Q&A pairs covering all 5 document categories.

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | **0.775** | 77.5% of claims grounded in retrieved docs |
| Answer Relevancy | **0.657** | 65.7% fully on-topic; prompt tightened post-eval |

Run evaluation locally:

```bash
python3 evaluate.py
# Results saved to evaluation_results.txt
```

> Evaluation uses the same Groq + HuggingFace stack. Requires `GROQ_API_KEY` in secrets.toml.

---

## Dependencies

```
streamlit              # UI framework
pypdf                  # PDF loading
langchain              # RAG orchestration
langchain-community    # PyPDFDirectoryLoader, FAISS
langchain-text-splitters
langchain-huggingface  # HuggingFaceEmbeddings
langchain-groq         # ChatGroq
sentence-transformers  # all-MiniLM-L6-v2 model
faiss-cpu              # Vector store
ragas                  # Evaluation metrics
```

---

## Documents Covered

| Source | Bills / Documents |
|---|---|
| Maryland Legislature (2026) | HB 992, SB 655 — "October 1st Cliff" e-waste producer law |
| Maryland Legislature (2026) | SB 553, HB 833 — Lithium-Ion Battery Safety Act |
| EPA (Feb 2026) | Universal Waste NPRM — proposed federal e-waste rule changes |
| Baltimore DPW | Disposal regulations + LWBB Task 3 operational protocols |
| NFPA 855 (2026) | Energy storage system safety standards |

---

## License

MIT
