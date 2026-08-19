"""
SpectrumLens — Pre-compute Embeddings for Instant Demo Load
============================================================
Loads all chunks from day1_chunks_output.json, embeds them via Jina API
in batches of 20 with 2s delays, and saves to disk for instant demo startup.

Usage:
    python precompute_embeddings.py              # uses Jina API
    python precompute_embeddings.py --provider hf  # uses HuggingFace Inference API
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "data/processed_chunks/day1_chunks_output.json"
NPZ_PATH = "data/precomputed_embeddings.npz"
PKL_PATH = "data/embedding_index.pkl"
BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 2.0


def load_chunks():
    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def embed_with_jina(texts):
    import requests
    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        raise EnvironmentError("JINA_API_KEY not set in .env")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    api_url = "https://api.jina.ai/v1/embeddings"
    model = "jina-embeddings-v5-text-small"

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  [{batch_num}/{total_batches}] Embedding {len(batch)} chunks ({i+1}-{min(i+BATCH_SIZE, total)}/{total})...")

        for attempt in range(5):
            try:
                resp = session.post(api_url, json={
                    "model": model,
                    "input": batch,
                    "dimensions": 1024,
                }, timeout=90)

                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                embeddings = [d["embedding"] for d in data.get("data", [])]
                all_embeddings.extend(embeddings)
                print(f"    OK ({len(embeddings)} embeddings)")
                break
            except Exception as e:
                if attempt < 4:
                    print(f"    Error (attempt {attempt+1}): {e}")
                    time.sleep(3 * (attempt + 1))
                else:
                    raise

        if i + BATCH_SIZE < total:
            print(f"    Waiting {DELAY_BETWEEN_BATCHES}s (rate limit)...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    return np.array(all_embeddings, dtype="float32")


def embed_with_hf(texts):
    import requests
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set in .env")

    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {hf_token}"})

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, 10):
        batch = texts[i:i + 10]
        batch_num = i // 10 + 1
        total_batches = (total + 9) // 10
        print(f"  [{batch_num}/{total_batches}] Embedding {len(batch)} chunks ({i+1}-{min(i+10, total)}/{total})...")

        for attempt in range(3):
            try:
                resp = session.post(api_url, json={"inputs": batch}, timeout=120)
                if resp.status_code == 503:
                    print("    Model loading, waiting 10s...")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        all_embeddings.extend(result)
                    else:
                        all_embeddings.append(result)
                print(f"    OK")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Error (attempt {attempt+1}): {e}")
                    time.sleep(3)
                else:
                    raise

        if i + 10 < total:
            time.sleep(0.2)

    arr = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-8)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for instant demo load")
    parser.add_argument("--provider", choices=["jina", "hf"], default="jina",
                        help="Embedding provider: 'jina' (Jina AI API) or 'hf' (HuggingFace Inference API)")
    args = parser.parse_args()

    print("=" * 60)
    print("SpectrumLens — Pre-compute Embeddings")
    print("=" * 60)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    texts = [c.get("normalized_text") or c.get("original_text") or c["text"] for c in chunks]
    print(f"Texts to embed: {len(texts)}")

    os.makedirs("data", exist_ok=True)

    t0 = time.perf_counter()

    if args.provider == "hf":
        print(f"\nUsing HuggingFace Inference API (BGE-M3)...")
        embeddings = embed_with_hf(texts)
    else:
        print(f"\nUsing Jina AI API (jina-embeddings-v5-text-small)...")
        embeddings = embed_with_jina(texts)

    elapsed = time.perf_counter() - t0
    print(f"\nEmbedded {len(embeddings)} chunks in {elapsed:.1f}s")

    np.savez_compressed(NPZ_PATH, embeddings=embeddings)
    print(f"Saved embeddings to {NPZ_PATH} ({os.path.getsize(NPZ_PATH) / 1024 / 1024:.1f} MB)")

    import pickle
    index_data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "model": "jina-embeddings-v5-text-small" if args.provider == "jina" else "BAAI/bge-m3",
        "provider": args.provider,
        "dim": embeddings.shape[1] if embeddings.ndim == 2 else 1024,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PKL_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"Saved index to {PKL_PATH} ({os.path.getsize(PKL_PATH) / 1024 / 1024:.1f} MB)")

    print(f"\nDone! Demo will now load instantly.")
    print(f"To rebuild: python precompute_embeddings.py")


if __name__ == "__main__":
    main()
