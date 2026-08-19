"""
Pre-compute embeddings for demo_app.py offline index.
Run once: python precompute_index.py
"""
import json, pickle, numpy as np, os, sys
from pathlib import Path

os.chdir(Path(__file__).parent)

CHUNKS_PATH   = "data/processed_chunks/day1_chunks_output.json"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT_NPZ       = "data/precomputed_embeddings.npz"
OUT_PKL       = "data/embedding_index.pkl"

print("=" * 60)
print("  SpectrumLens — Pre-compute Embedding Index")
print("=" * 60)

# Load chunks
print(f"\n[1/3] Loading chunks from {CHUNKS_PATH}...")
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"      {len(chunks)} chunks loaded")

# Load model
print(f"\n[2/3] Loading {EMBED_MODEL}...")
print("      (First run downloads ~490 MB — subsequent runs use cache)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(EMBED_MODEL)
print("      Model ready")

# Encode
print(f"\n[3/3] Encoding {len(chunks)} chunks...")
texts = [c.get("normalized_text") or c.get("text", "") for c in chunks]
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)
arr = np.array(embeddings, dtype="float32")
print(f"      Shape: {arr.shape}  dtype: {arr.dtype}")

# Save
np.savez_compressed(OUT_NPZ, embeddings=arr)
with open(OUT_PKL, "wb") as f:
    pickle.dump({"chunks": chunks, "model": EMBED_MODEL}, f)

size_mb = Path(OUT_NPZ).stat().st_size / 1024 / 1024
print(f"\n✅  Saved {OUT_NPZ} ({size_mb:.1f} MB)")
print(f"✅  Saved {OUT_PKL}")
print("\n  Next 'streamlit run demo_app.py' will load instantly (no model download).")
print("=" * 60)
