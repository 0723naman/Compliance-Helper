# scripts/reindex_twitter_complete.py
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ingest import index_documents
from config import config

FILE_PATH = "data\uploaded\Terms of Service Twitter.pdf"  # your uploaded file path

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"{FILE_PATH} not found. Please upload it and retry.")

print("Starting ingestion for:", FILE_PATH)
index, meta = index_documents([FILE_PATH], save_index=True, debug=True)
print(f"Indexed {len(meta)} chunks.")
print("Wrote index to:", config.VECTOR_STORE_PATH)
print("Wrote metadata to:", config.METADATA_PATH)
print("Debug file at:", Path(config.VECTOR_STORE_PATH).parent / "ingest_debug.json")
