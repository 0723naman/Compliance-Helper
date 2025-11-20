# utils/response_formatter.py
from typing import List, Dict

BASE_SYSTEM_INSTRUCTION = (
    "You are the company's Compliance Helper assistant. Answer employee questions using ONLY the provided policy context. "
    "Always cite the source(s) you used inline in square brackets using the format [FILENAME#chunk_id]. "
    "If you cannot find a definitive answer in the provided policies, say exactly: "
    "'I couldn't find a definitive answer in the provided policies.' "
    "Then suggest who to contact (for example: HR) or next steps to get the official guidance."
)

def build_system_prompt(retrieved: List[Dict], max_chars_per_snippet: int = 1200) -> str:
    if not retrieved:
        context_block = "No policy context found."
    else:
        parts = []
        for r in retrieved:
            label = f"[{r['doc_id']}#{r['chunk_id']}]"
            snippet = (r.get("text") or "").strip()
            if len(snippet) > max_chars_per_snippet:
                snippet = snippet[:max_chars_per_snippet] + " ... (truncated)"
            parts.append(f"{label}\n{snippet}")
        context_block = "\n\n---\n\n".join(parts)

    prompt = f"{BASE_SYSTEM_INSTRUCTION}\n\nContext:\n{context_block}\n\nWhen you answer, cite the source labels inline where appropriate."
    return prompt
