````markdown
# Compliance Helper — AI-Powered Internal Policy Assistant

The **Compliance Helper** is an AI-powered chatbot that helps employees quickly find accurate answers from internal company policies, guidelines, and compliance documents. It uses a Retrieval-Augmented Generation (RAG) architecture built with **Gemini 1.5 Flash**, **FAISS**, and a **Streamlit** user interface.

This project is designed to streamline policy discovery, reduce compliance uncertainty, and enable organizations to provide reliable support for internal queries.

---

## Key Features

### High-Accuracy RAG Pipeline
- Extracts text from PDF/TXT policy documents  
- Cleans and preprocesses the content  
- Sentence-level chunking (1200 tokens + 200 overlap)  
- Embedding generation using Gemini 1.5 Flash  
- Vector search with FAISS (Inner Product index)  
- Retrieves relevant policy chunks and provides answers with citations  

### Streamlit Chat Interface
- Upload policy documents  
- Build a custom vector index  
- Ask compliance-related questions  
- Receive contextual answers grounded in your documents  

### Document Ingestion Pipeline
- PDF text extraction via `PyPDF2`  
- Cleaning and normalizing long-form policy documents  
- Chunking logic designed for compliance/legal content  
- Batch embedding + normalization for FAISS  

### Safe & Secure
- `.env` used for storing API keys  
- No raw PDF content is sent to Gemini — only embeddings  
- `venv/` excluded from repository  

---

## Why This Project Is Useful

Organizations often store critical compliance documents in long and complex PDFs. Employees struggle to locate the exact clause or rule they need, leading to:

- Misinterpretation of policies  
- Repeated dependency on HR/legal teams  
- Slow onboarding  
- Increased compliance risk  

**Compliance Helper solves this by enabling fast, accurate, citation-backed responses based on internal policies.**

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/0723naman/Compliance-Helper.git
cd Compliance-Helper
````

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Set up environment variables

Create `.env` in the project root:

```
GEMINI_API_KEY=YOUR_API_KEY
LLM_MODEL=gemini-1.5-flash
EMBED_MODEL=embedding-001
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

---

## 5. Build the Vector Index

Place your PDF/TXT files inside the project or note their file path.

Update the ingestion script:

```
scripts/reindex_twitter_complete.py
```

Run the indexing script:

```bash
python scripts/reindex_twitter_complete.py
```

This generates:

* `data/faiss.index`
* `data/metadata.json`
* `data/ingest_debug.json`

---

## 6. Run the Chatbot

```bash
streamlit run app.py
```

Open the browser UI:

* Upload documents
* Build the index
* Ask your questions

---

## Usage Example

Ask questions like:

```
What are the user data retention rules?
Who is allowed to access internal logs?
What is the policy for third-party data sharing?
```

Responses include:

* Grounded answers
* Policy citations
* Chunk references (e.g., `[Policy.pdf#12]`)

---

## Project Structure

```
AI_UseCase/
│
├── app.py                      # Streamlit UI
├── config/
│   └── config.py               # Settings & environment variables
│
├── models/
│   ├── llm.py                  # Gemini chat model wrapper
│   └── embeddings.py           # Gemini embedding generator
│
├── utils/
│   ├── ingest.py               # PDF ingestion, cleaning, chunking, FAISS building
│   ├── retriever.py            # Vector search retriever
│   └── response_formatter.py   # RAG system prompt + citation formatting
│
├── scripts/
│   ├── reindex_twitter_complete.py
│   └── test_embeddings.py
│
├── data/                       # Auto-generated index + metadata
│
├── requirements.txt
└── .env
```

Each module is modular, testable, and designed to be extended for new document types or retrieval strategies.

---

## Getting Help

For support:

* Open an issue:
  [https://github.com/0723naman/Compliance-Helper/issues](https://github.com/0723naman/Compliance-Helper/issues)

* Troubleshooting topics include:

  * Embedding failures
  * Chunking quality
  * FAISS index errors
  * API limitations

---

## Maintainers & Contributions

Maintainer: **Naman Raj Yadav**

Contributions are welcome.
Before submitting, please:

* Open an issue for discussion
* Follow best practices for PRs
* Avoid committing sensitive files (`.env`, `venv/`, policy PDFs)

Future improvements will include:

* BM25 hybrid search
* ColBERT re-ranking
* Admin dashboards
* Multi-tenant indexing

---

## License

This project uses the MIT License.
See the `LICENSE` file for more details.

```

---

## Want a more advanced README?

I can generate:

- Badge-rich GitHub README  
- README with project diagrams + architecture diagram (ASCII or image)  
- README with inline GIF demos  
- README optimized for HR visibility (portfolio-style)

Just tell me.
```
