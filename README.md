```markdown
# Compliance Helper — AI-Powered Internal Policy Assistant

The Compliance Helper is an AI-powered chatbot designed to help employees quickly find answers inside internal company policies, procedures, and compliance documents. It uses a Retrieval-Augmented Generation (RAG) pipeline powered by Gemini 1.5 Flash, FAISS vector search, and Streamlit.

---

## Features

- Upload internal policy PDFs or TXT files
- Automatic text cleaning, extraction, and sentence-aware chunking
- Gemini embeddings + FAISS index creation
- RAG-based chatbot with high-accuracy compliance responses
- Inline citations pointing to exact policy chunks
- Multi-document support
- Streamlit-based user interface
- Custom compliance-focused system prompt

---

## System Architecture

```
PDF/TXT Files → Text Extraction → Cleaning → Chunking → Gemini Embeddings → FAISS Index  
                                   ↑                                 ↓  
                                   └──────── Retrieval (Top Chunks) ─┘  
                                               ↓  
                                     Gemini 1.5 Flash  
                                               ↓  
                                   Final Answer With Citations
```

---

## Folder Structure

```
AI_UseCase/
│
├── app.py                      # Streamlit chatbot
├── config/
│   └── config.py               # Environment variables & constants
│
├── models/
│   ├── llm.py                  # Gemini chat model
│   └── embeddings.py           # Gemini embeddings handler
│
├── utils/
│   ├── ingest.py               # PDF parsing, chunking, FAISS ingestion
│   ├── retriever.py            # Vector store search
│   └── response_formatter.py   # RAG system prompt & output formatting
│
├── scripts/
│   ├── reindex_twitter_complete.py
│   └── test_embeddings.py
│
├── data/                       # FAISS index + metadata stored here
│
├── requirements.txt
└── .env
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/0723naman/Compliance-Helper.git
cd Compliance-Helper
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create `.env` file:

```
GEMINI_API_KEY=YOUR_API_KEY
LLM_MODEL=gemini-1.5-flash
EMBED_MODEL=embedding-001
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

---

## Build the Vector Index

Update the file path inside:

```
scripts/reindex_twitter_complete.py
```

Then run:

```bash
python scripts/reindex_twitter_complete.py
```

Generated files:

* data/faiss.index
* data/metadata.json
* data/ingest_debug.json

---

## Run the Chatbot

```bash
streamlit run app.py
```

You can now:

* Upload policy files
* Build vector index
* Ask compliance questions
* Receive citations from the policy documents

---

## Security Notes

* `.env` is git-ignored (API key safe)
* `venv/` folder is git-ignored
* Only embeddings are sent to Gemini (documents remain local)
* No compliance PDF is uploaded to external services

---

## Future Improvements

* Hybrid search (BM25 + embeddings)
* Re-ranker integration (BGE/ColBERT)
* Admin analytics dashboard
* Multi-tenant support
* Automated PDF ingestion pipeline

---

## Contributing

Contributions are welcome. Open an issue for discussion before major changes.

---

## License

MIT License.
```
