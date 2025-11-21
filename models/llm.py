from typing import List
from config import config

class GeminiChatWrapper:
    """
    Wraps Google GenAI / Gemini model to expose .invoke(messages)
    where messages is a list of langchain_core message objects (SystemMessage/HumanMessage/AIMessage).
    """

    def __init__(self):
        try:
            from google import genai
        except Exception as e:
            raise RuntimeError("google-genai package required. pip install google-genai") from e

        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set in environment.")

        # instantiate client (reads api_key param)
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model = config.LLM_MODEL

    def invoke(self, messages: List[object]):
        """
        Accepts list of langchain message objects. Converts to a single concatenated prompt.
        Returns an object with .content attribute (string reply).
        """
        # Build a prompt: system + conversation history
        prompt_parts = []
        for m in messages:
            cls_name = m.__class__.__name__.lower()
            if "system" in cls_name:
                prompt_parts.append(f"[SYSTEM]\n{m.content}\n")
            elif "human" in cls_name:
                prompt_parts.append(f"[USER]\n{m.content}\n")
            elif "ai" in cls_name:
                prompt_parts.append(f"[ASSISTANT]\n{m.content}\n")
            else:
                # fallback to content only
                prompt_parts.append(m.content)

        # Join into a single prompt; Gemini works well with a system instruction followed by user input.
        full_prompt = "\n\n".join(prompt_parts)

        # Call Gemini generate_content
        # The GenAI SDK provides client.models.generate_content(model=..., contents=...)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )

        # The response may be accessible via resp.text or resp.output or resp.result.
        # Common pattern: resp.text or resp.output[0].content[0].text — adapt robustly:
        text = None
        # Try known attributes:
        if hasattr(resp, "text") and isinstance(resp.text, str):
            text = resp.text
        else:
            # Some SDKs return resp.output[0].content[0].text
            try:
                out = getattr(resp, "output", None)
                if out and isinstance(out, (list, tuple)) and len(out) > 0:
                    # content elements
                    first = out[0]
                    # some shapes: first['content'][0]['text'] or first.content[0].text
                    if isinstance(first, dict):
                        # dict style
                        content = first.get("content")
                        if content and isinstance(content, (list, tuple)) and len(content) > 0:
                            maybe = content[0]
                            text = maybe.get("text") or maybe.get("payload") or str(maybe)
                    else:
                        # try attribute access
                        content = getattr(first, "content", None)
                        if content and isinstance(content, (list, tuple)) and len(content) > 0:
                            text = getattr(content[0], "text", None)
            except Exception:
                text = None

        if text is None:
            # last resort, stringify whole response
            text = str(resp)

        # Return object with .content to match template usage
        class R:
            def __init__(self, content):
                self.content = content
        return R(text)


def get_chatgroq_model():
    """
    Factory function used in your app template.
    """
    return GeminiChatWrapper()

