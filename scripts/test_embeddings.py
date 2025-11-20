# scripts/test_embeddings.py
import os, sys
from pathlib import Path

# Ensure project root is on sys.path so 'models' imports the local package, not a PyPI package.
# This makes the script runnable even if the environment's current working directory differs.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("Project root added to sys.path:", str(project_root))

# Now import the local embeddings module
from models.embeddings import embed_texts

texts = [
    "Test embedding one.",
    "Another short test for embeddings."
]

try:
    embs = embed_texts(texts)
    print(f"SUCCESS: Got {len(embs)} embeddings. Dim of first vector: {len(embs[0])}")
    print("First embedding (first 8 dims):", embs[0][:8])
except Exception as e:
    print("Embedding test failed:", repr(e))
