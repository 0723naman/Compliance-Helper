# models/embeddings.py
from typing import List, Any
from config import config
import json

def _get_genai_client():
    try:
        from google import genai
    except Exception as e:
        raise RuntimeError(
            "google-genai not installed. Install with: pip install google-genai"
        ) from e

    if not config.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return client

def _short_repr(obj: Any, length: int = 1500) -> str:
    try:
        s = json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False)
    except Exception:
        s = repr(obj)
    return (s[:length] + "...(truncated)") if len(s) > length else s

def _extract_from_item(item: Any) -> List[float]:
    """
    Try to extract a float vector from a single embedding item in various shapes.
    """
    # dict-like: "embedding"
    if isinstance(item, dict):
        if "embedding" in item and isinstance(item["embedding"], (list, tuple)):
            return list(item["embedding"])
        if "values" in item and isinstance(item["values"], (list, tuple)):
            # Gemini sometimes uses "values"
            return list(item["values"])
        # some SDKs use nested fields: item["data"][0]["embedding"] etc. handle later globally
    # object-like with attribute .embedding or .values
    emb_attr = getattr(item, "embedding", None)
    if emb_attr is not None and isinstance(emb_attr, (list, tuple)):
        return list(emb_attr)
    values_attr = getattr(item, "values", None)
    if values_attr is not None and isinstance(values_attr, (list, tuple)):
        return list(values_attr)
    # list/tuple itself
    if isinstance(item, (list, tuple)) and all(isinstance(x, (float, int)) for x in item):
        return list(item)
    # if none matched, return empty to signal failure for this item
    return []

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for the provided texts using Google GenAI (Gemini).
    Parses several common response shapes including ones with 'embeddings' -> [{'values': [...]}].
    Raises a helpful RuntimeError including a truncated raw response if it cannot parse.
    """
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list of strings")

    client = _get_genai_client()
    model = config.EMBEDDING_MODEL

    # Call embeddings API
    try:
        resp = client.models.embed_content(model=model, contents=texts)
    except Exception as e:
        raise RuntimeError(f"Embedding API call failed: {e}")

    # Try many known shapes, in priority order.

    # 1) resp.embeddings -> list of items where item has "values" or "embedding"
    try:
        if hasattr(resp, "embeddings") and isinstance(resp.embeddings, (list, tuple)):
            out = []
            for item in resp.embeddings:
                vec = _extract_from_item(item)
                if not vec:
                    # try deeper dict access
                    if isinstance(item, dict):
                        # try item.get("data") or item.get("embedding") etc.
                        if "data" in item and isinstance(item["data"], (list, tuple)):
                            # look into data elements
                            found = False
                            for d in item["data"]:
                                vec2 = _extract_from_item(d)
                                if vec2:
                                    out.append(vec2); found = True; break
                            if found:
                                continue
                    # fallback: empty, will handle below
                else:
                    out.append(vec)
            if out and len(out) == len(texts):
                return out
            # if lengths mismatch but there are some, return them if count matches texts; otherwise continue trying
            if out and len(out) >= 1 and len(out) == len(resp.embeddings):
                return out
    except Exception:
        pass

    # 2) resp.output or resp.responses or resp.data top-level
    try:
        for candidate_name in ("output", "responses", "data", "embeds", "results"):
            out_block = getattr(resp, candidate_name, None) if hasattr(resp, candidate_name) else (resp.get(candidate_name) if isinstance(resp, dict) else None)
            if out_block and isinstance(out_block, (list, tuple)):
                out = []
                for el in out_block:
                    # el may be dict or object
                    vec = _extract_from_item(el)
                    if vec:
                        out.append(vec); continue
                    # nested possibilities
                    if isinstance(el, dict):
                        # look for embedding in nested places
                        for key in ("embedding", "embeddings", "values", "embeddingVector", "vector"):
                            v = el.get(key)
                            if isinstance(v, (list, tuple)):
                                out.append(list(v)); break
                        else:
                            # try nested 'content' -> list -> dict with 'embedding' or 'values'
                            cont = el.get("content") or el.get("outputs") or el.get("result")
                            if cont and isinstance(cont, (list, tuple)):
                                found = False
                                for c in cont:
                                    vec2 = _extract_from_item(c)
                                    if vec2:
                                        out.append(vec2); found = True; break
                                if not found:
                                    # attempt to find embedding in deeper dicts
                                    for c in cont:
                                        if isinstance(c, dict):
                                            for key in ("embedding", "values"):
                                                if key in c and isinstance(c[key], (list, tuple)):
                                                    out.append(list(c[key])); found = True; break
                                        if found: break
                            # else not found in this element
                    else:
                        # object-like: check attributes
                        vec2 = _extract_from_item(el)
                        if vec2:
                            out.append(vec2)
                if out and len(out) == len(texts):
                    return out
    except Exception:
        pass

    # 3) resp is dict with "embeddings" key (dict style)
    try:
        if isinstance(resp, dict) and "embeddings" in resp:
            arr = resp["embeddings"]
            if isinstance(arr, (list, tuple)):
                out = []
                for item in arr:
                    vec = _extract_from_item(item)
                    if vec:
                        out.append(vec)
                if out and len(out) == len(texts):
                    return out
    except Exception:
        pass

    # 4) resp may be an object whose __dict__ contains embeddings-like keys
    try:
        attrs = getattr(resp, "__dict__", None)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if k.lower().startswith("embed") and isinstance(v, (list, tuple)):
                    out = []
                    for item in v:
                        vec = _extract_from_item(item)
                        if vec:
                            out.append(vec)
                    if out and len(out) == len(texts):
                        return out
    except Exception:
        pass

    # If we reach here, we couldn't parse a matching vector list.
    raw = _short_repr(resp)
    raise RuntimeError(
        "Unexpected embeddings response shape. Could not parse embeddings from the Gemini response.\n\n"
        "Truncated raw response (for debugging):\n\n"
        f"{raw}\n\n"
        "If you see an unfamiliar structure, paste the truncated response here and I will adapt the parser."
    )
