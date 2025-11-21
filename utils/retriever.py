import os
import json
from config import config
from models.embeddings import embed_texts

def load_index_and_meta():
    try:
        import faiss
    except Exception:
        raise RuntimeError("faiss-cpu required.")
    if not (os.path.exists(config.VECTOR_STORE_PATH) and os.path.exists(config.METADATA_PATH)):
        raise FileNotFoundError("Vector store or metadata missing. Build the index first.")
    index = faiss.read_index(config.VECTOR_STORE_PATH)
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def _keyword_score(query: str, text: str) -> int:
    # Simple lexical scoring: count overlapping words (case-insensitive)
    import re
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))
    return len(q_words & t_words)

def retrieve(query: str, index, metadata, k: int = None, lexical_k: int = 10, sim_threshold: float = 0.18):
    """
    Returns list of items:
    { score: float, doc_id: str, chunk_id: int, text: str }
    """
    import numpy as np
    import faiss
    k = k or config.MAX_RETRIEVALS
    q_emb = embed_texts([query])[0]
    q_arr = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q_arr)
    D, I = index.search(q_arr, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "doc_id": m.get("doc_id"),
            "chunk_id": m.get("chunk_id"),
            "text": m.get("text")
        })

    # If top result(s) are below sim_threshold, do a lexical fallback and merge results
    top_score = results[0]["score"] if results else 0.0
    if top_score < sim_threshold:
        # compute lexical scores over all metadata (fast enough for moderate sizes)
        lex_scores = []
        for m in metadata:
            sc = _keyword_score(query, m.get("text",""))
            if sc > 0:
                lex_scores.append((sc, m))
        # sort lexical matches by score
        lex_scores.sort(key=lambda x: x[0], reverse=True)
        # take top lexical_k and add into results if not already present
        added = 0
        existing_keys = {(r["doc_id"], r["chunk_id"]) for r in results}
        for sc, m in lex_scores[:lexical_k]:
            key = (m.get("doc_id"), m.get("chunk_id"))
            if key in existing_keys:
                continue
            results.append({
                "score": float(0.0 + sc/100.0),  # lexical pseudo-score
                "doc_id": m.get("doc_id"),
                "chunk_id": m.get("chunk_id"),
                "text": m.get("text")
            })
            added += 1
            if added >= lexical_k:
                break

    # final: sort results by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    # return up to k results (but allow more for inspection)
    return results[: max(k, len(results))]

