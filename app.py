import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from models.llm import get_chatgroq_model
from utils.response_formatter import build_system_prompt
from utils.retriever import load_index_and_meta, retrieve
from utils.ingest import index_documents
from config import config
from pathlib import Path

def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    st.title("The Chatbot Blueprint (Gemini)")
    st.markdown("This app uses Google Gemini (Gemini models) via the google-genai Python SDK.")
    st.markdown("""
    Steps:
    1. Set GOOGLE_API_KEY in environment (Gemini API key from AI Studio).
    2. Upload internal policy PDFs/TXT in the Chat page sidebar and click 'Build index'.
    3. Ask employee questions — the assistant uses retrieved policy snippets and provides citations.
    """)
    st.markdown("---")

def chat_page():
    st.title("🤖 Compliance Helper — Gemini (Internal Policies)")

    # Initialize chat model
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        chat_model = None
        st.warning(f"LLM initialization error: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for ingestion and settings
    with st.sidebar:
        st.header("Index / Documents")
        uploaded = st.file_uploader("Upload policy PDF/TXT files (multiple)", accept_multiple_files=True, type=["pdf","txt"])
        if st.button("Build index from uploaded files"):
            if not uploaded:
                st.sidebar.error("No files uploaded.")
            else:
                tmp_dir = Path("data/uploaded")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_paths = []
                for f in uploaded:
                    out_path = tmp_dir / f.name
                    with open(out_path, "wb") as fh:
                        fh.write(f.read())
                    tmp_paths.append(str(out_path))
                try:
                    with st.spinner("Indexing documents (may take a while)..."):
                        idx, meta = index_documents(tmp_paths, save_index=True)
                    st.sidebar.success(f"Indexed {len(meta)} chunks from {len(tmp_paths)} files.")
                except Exception as e:
                    st.sidebar.error(f"Ingestion failed: {e}")

        st.markdown("---")
        st.header("Settings")
        mode = st.radio("Response mode", ["concise", "detailed"], index=0)
        max_k = st.slider("Retrieved snippets (k)", min_value=1, max_value=12, value=config.MAX_RETRIEVALS)

        st.markdown("---")
        if st.button("Clear chat history in this session"):
            st.session_state.messages = []
            st.experimental_rerun()

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Type your message here (employee question)...")

    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ensure index exists
        try:
            index, meta = load_index_and_meta()
        except Exception as e:
            st.error("Vector index not found. Upload policy documents in the sidebar and build index first.")
            return

        # Retrieve
        with st.spinner("Retrieving relevant policy snippets..."):
            try:
                retrieved = retrieve(prompt, index, meta, k=max_k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                retrieved = []

        # Build system prompt from retrieved
        system_prompt = build_system_prompt(retrieved)

        # Get response from model
        with st.chat_message("assistant"):
            with st.spinner("Generating answer with Gemini..."):
                if not chat_model:
                    st.error("LLM not available. Check GOOGLE_API_KEY and model config.")
                    return
                response_text = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response_text)

        st.session_state.messages.append({"role":"assistant","content":response_text})

        # Show retrieved snippets
        st.markdown("---")
        st.subheader("Cited policy snippets")
        if not retrieved:
            st.write("No policy snippets were retrieved.")
        else:
            for r in retrieved:
                st.markdown(f"**{r.get('doc_id')}#{r.get('chunk_id')}** — similarity: {r.get('score'):.3f}")
                snippet = r.get("text","")
                st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))

def main():
    st.set_page_config(page_title="Gemini Compliance Helper", page_icon="🤖", layout="wide")
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()

