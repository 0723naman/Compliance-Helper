import os
import json
import re
from typing import List, Tuple
from pathlib import Path
from config import config
from models.embeddings import embed_texts

# Helper: try to use tiktoken for token-accurate chunking if available
def _get_tokenizer():
    try:
        import tiktoken
        # choose encoder; fallback to cl100k_base which works for many models
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("gpt2")
        return enc
    except Exception:
        return None

TOKEN_ENCODER = _get_tokenizer()

def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from PDF using pdfplumber if available (best), otherwise PyPDF2 fallback.
    Returns cleaned full-text string (no page breaks removed intentionally yet).
    """
    path = str(path)
    text_pages = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text_pages.append(txt)
    except Exception:
        # fallback
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    text_pages.append(txt)
        except Exception as e:
            raise RuntimeError("Unable to read PDF. Install pdfplumber or PyPDF2.") from e

    # text_pages is list of page strings; keep per-page for header/footer detection
    # Basic cleaning: normalize whitespace
    pages_clean = []
    for p in text_pages:
        if not p:
            pages_clean.append("")
            continue
        s = p.replace('\r\n', '\n').replace('\r', '\n')
        s = re.sub(r'\n{3,}', '\n\n', s)
        pages_clean.append(s.strip())

    full = "\n\n".join(pages_clean)
    # We will clean repeated headers/footers next in _remove_repeated_headers
    full = _remove_repeated_headers(pages_clean)
    full = _fix_hyphenation_and_broken_lines(full)
    return full

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = _fix_hyphenation_and_broken_lines(raw)
    return raw

def _remove_repeated_headers(pages: List[str]) -> str:
    """
    Heuristic: find short lines that repeat across many pages and remove them.
    Input: list of per-page strings.
    Returns concatenated cleaned text.
    """
    # gather candidate lines across pages
    candidate_counts = {}
    page_lines = []
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        page_lines.append(lines)
        # only consider short lines (likely headers)
        for ln in lines:
            if 3 <= len(ln) <= 120:
                candidate_counts[ln] = candidate_counts.get(ln, 0) + 1

    repeated = {ln for ln, c in candidate_counts.items() if c >= max(2, len(pages)//6)}  # appears on multiple pages
    # create regex to remove repeated lines (escape)
    if repeated:
        pattern = re.compile("|".join(re.escape(r) for r in sorted(repeated, key=len, reverse=True)))
    else:
        pattern = None

    cleaned_pages = []
    for lines in page_lines:
        if not lines:
            cleaned_pages.append("")
            continue
        if pattern:
            new_lines = []
            for ln in lines:
                if pattern.search(ln):
                    # skip header/footer lines
                    continue
                new_lines.append(ln)
            cleaned_pages.append("\n".join(new_lines))
        else:
            cleaned_pages.append("\n".join(lines))

    return "\n\n".join(cleaned_pages)

def _fix_hyphenation_and_broken_lines(text: str) -> str:
    # remove hyphenation at line breaks e.g. "exam-\nple" -> "example"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # join broken sentence lines: if a line break occurs without sentence end, join lines
    lines = text.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if i + 1 < len(lines):
            nxt = lines[i+1].lstrip()
            # if ln doesn't end with punctuation (or ends with hyphen handled above) and next starts with lowercase or a number, join them
            if ln and not re.search(r'[.!?:"\)\]]\s*$', ln) and nxt and (nxt[0].islower() or nxt[0].isdigit()):
                ln = ln + " " + nxt
                i += 2
                out_lines.append(ln)
                continue
        out_lines.append(ln)
        i += 1
    # normalize multiple blank lines
    s = "\n".join(out_lines)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

# Sentence splitter (regex-based; conservative)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a conservative regex that avoids variable-width lookbehinds.
    Returns a list of sentence strings.
    """
    if not text:
        return []
    # Normalize whitespace first
    t = re.sub(r'\s+', ' ', text).strip()
    # Split on our pattern
    parts = _SENTENCE_SPLIT_RE.split(t)
    # Fallback: if nothing split (very short text), do naive split by period-space
    if len(parts) == 1:
        parts = re.split(r'(?<=[\.\?\!])\s+', t)
    # Trim and return
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def _count_tokens(text: str) -> int:
    """
    If tiktoken is available, use it for accurate token counts; otherwise fall back to word count.
    """
    if TOKEN_ENCODER:
        try:
            return len(TOKEN_ENCODER.encode(text))
        except Exception:
            pass
    # fallback: approximate tokens ~= words
    return len(text.split())

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Chunk text by accumulating sentences until an approximate token count >= chunk_size is reached.
    chunk_size and overlap are interpreted as token counts (or word counts if tiktoken not available).
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    N = len(sentences)
    # Precompute sentence token counts
    sent_tokens = [_count_tokens(s) for s in sentences]

    while i < N:
        token_acc = 0
        chunk_sentences = []
        j = i
        while j < N and token_acc < chunk_size:
            chunk_sentences.append(sentences[j])
            token_acc += sent_tokens[j]
            j += 1
        # join sentences to form chunk
        chunk_text = " ".join(chunk_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)
        # advance i with overlap
        # compute how many sentences to step: find new start such that overlap tokens are preserved
        if j >= N:
            break
        # Determine how many sentences to step back to create overlap in tokens
        # We want next_start = j - k where sum(sent_tokens[j-k : j]) >= overlap
        if overlap <= 0:
            next_start = j
        else:
            k = 0
            ov_acc = 0
            while (j - 1 - k) >= 0 and ov_acc < overlap:
                ov_acc += sent_tokens[j - 1 - k]
                k += 1
                if (j - k) <= i:
                    # don't move backwards beyond current i
                    break
            next_start = max(i + 1, j - k)  # ensure progress by at least 1 sentence
        i = next_start

    return chunks

def index_documents(file_paths: List[str], save_index: bool = True, debug: bool = True):
    """
    Ingest PDF/TXT files, chunk using sentence-accumulation with token-awareness, compute embeddings,
    build FAISS index and write metadata. Returns (index, metadata).
    If debug=True writes a debug JSON file with stats and first/last chunk examples.
    """
    try:
        import faiss
        import numpy as np
    except Exception:
        raise RuntimeError("faiss-cpu and numpy required. pip install faiss-cpu numpy")

    docs = []
    for p in file_paths:
        p = str(p)
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(p)
        else:
            text = extract_text_from_txt(p)
        docs.append((Path(p).name, text))

    all_chunks = []
    metadata = []
    stats = {"files": [], "total_sentences": 0, "total_words": 0, "total_chunks": 0}

    for doc_name, text in docs:
        sentences = _split_into_sentences(text)
        words = len(text.split())
        stats["files"].append({"doc_name": doc_name, "sentences": len(sentences), "words": words})
        stats["total_sentences"] += len(sentences)
        stats["total_words"] += words

        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            metadata.append({"doc_id": doc_name, "chunk_id": idx, "text": c})
            all_chunks.append(c)

    stats["total_chunks"] = len(all_chunks)

    if not all_chunks:
        raise RuntimeError("No text extracted from provided documents.")

    # embeddings in batches
    batch_size = 20
    embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_embs = embed_texts(batch_texts)
        embeddings.extend(batch_embs)

    xb = np.array(embeddings).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    if save_index:
        os.makedirs(Path(config.VECTOR_STORE_PATH).parent, exist_ok=True)
        faiss.write_index(index, config.VECTOR_STORE_PATH)
        with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        if debug:
            debug_info = {
                "stats": stats,
                "sample_first_chunk": all_chunks[0] if all_chunks else "",
                "sample_last_chunk": all_chunks[-1] if all_chunks else "",
                "total_chunks": len(all_chunks)
            }
            with open(Path(config.VECTOR_STORE_PATH).parent / "ingest_debug.json", "w", encoding="utf-8") as df:
                json.dump(debug_info, df, ensure_ascii=False, indent=2)

    return index, metadata

