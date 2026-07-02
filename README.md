# Sensitive RAG + Compliance

A Streamlit-based document intelligence app for uploading PDFs, TXT, and CSV files, extracting content, indexing it for retrieval, detecting sensitive or compliance-relevant findings, and answering questions over the uploaded documents.

## Setup Instructions

### 1. Prerequisites

- Python 3.13+
- `pip` or `uv`
- An OpenRouter or OpenAI-compatible API key if you want LLM-powered detection and summaries

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you use `uv`, you can also install from `pyproject.toml` / `uv.lock`.

### 4. Configure environment variables

Copy the example file and fill in your values:

```bash
copy .env.example .env
```

Minimum recommended variables:

```env
ENABLE_LLM=1
LLM_PROVIDER=langchain
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
LLM_APP_NAME=Sensitive RAG + Compliance
LLM_APP_URL=http://localhost:8501
SESSION_TTL_MINUTES=10
```

### 5. Run the app

```bash
streamlit run app/main.py
```

The app will start the Streamlit UI and create local storage automatically.


## Architecture Overview

The app follows a document-processing pipeline:

1. **Upload layer**
   - Users upload PDF, TXT, or CSV files from the Streamlit UI.
   - Files are stored in a session-scoped temporary directory.

2. **Parsing layer**
   - PDFs are parsed with PyMuPDF.
   - TXT and CSV files are normalized into page-like text units.

3. **Chunking and indexing layer**
   - Parsed text is split into chunks.
   - Chunks are embedded with a sentence-transformer model.
   - The app indexes content in both:
     - **ChromaDB** for vector retrieval
     - **BM25** for keyword retrieval

4. **Detection layer**
   - Regex and heuristic detection identify high-confidence patterns such as emails, Aadhaar-like numbers, phone numbers, API keys, passwords, employee IDs, bank details, and confidential business terms.
   - spaCy NER is used for contextual entities where available.
   - An LLM review step examines the detected findings with surrounding document context when enabled.
   - Findings are deduplicated and then labeled as `regex` or `regex+LLM` in the UI depending on whether the LLM path was used.

5. **Compliance layer**
   - Findings are scored into a simple risk bucket.
   - The dashboard summarizes documents and shows detected findings per file.

6. **QA layer**
   - A hybrid retriever combines BM25 and vector search.
   - Questions are answered over retrieved context, with special handling for entity-count style questions.

7. **Session and cleanup layer**
   - Session data is stored in SQLite.
   - Temporary files are scoped to the session.
   - A cleanup endpoint and session-expiry flow remove old artifacts.

## AI/ML Approach Used

This project uses a hybrid AI/ML pipeline rather than a single model:

- **Sentence-transformer embeddings** for semantic similarity search
- **BM25** for lexical search
- **ChromaDB** as the vector store
- **spaCy NER** for contextual entity extraction
- **Regex + heuristics** for high-precision pattern detection
- **LLM review via OpenRouter/OpenAI-compatible API** for:
  - reviewing detected findings with surrounding context
  - producing a short compliance summary

The detection strategy is intentionally layered:

1. Deterministic patterns find likely sensitive content.
2. The LLM reviews the detected findings with surrounding document context when enabled.
3. The final findings are merged, deduplicated, and stored for the dashboard and QA flow.

## Challenges Faced

- **False positives from numeric patterns**
  - Phone numbers, account numbers, and IDs can look similar.
  - The app uses surrounding context and LLM review to reduce misclassification.

- **Placeholder and example detection**
  - Synthetic values such as demo numbers or sample secrets can look valid.
  - The pipeline relies on context-aware LLM review plus deterministic filtering to reduce these cases.

- **Repeated results from multiple detectors**
  - Regex and LLM can detect the same item in different ways.
  - Findings are deduplicated before saving and display.

- **LLM availability and latency**
  - LLM calls can fail or be slow depending on the provider.
  - The app falls back to regex/NER when needed and reports the failure reason.

- **Session and storage cleanup**
  - Temporary files, SQLite rows, and vector-store state need to stay session-scoped.
  - The app includes cleanup logic so stale data does not accumulate indefinitely.

## Future Improvements

- Add a stronger OCR path for scanned or image-only PDFs
- Improve evaluation with a labeled test set for precision/recall measurement
- Add background jobs for indexing to reduce UI wait time
- Make per-entity explanations richer in the dashboard
- Add user-configurable retention windows for session cleanup
- Expand compliance mapping and reporting views
- Add a small benchmark page for comparing regex-only vs hybrid vs LLM-assisted detection

## Working Prototype Deployment Link

- Local prototype: http://localhost:8501

