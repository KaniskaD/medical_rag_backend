import os
import json
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = os.path.join("faiss_indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

# Text model: 384D
TEXT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
TEXT_DIM = TEXT_MODEL.get_sentence_embedding_dimension()  # typically 384

# Image model dim produced by CLIP (utils_image uses clip-ViT-B-32) -> 512
IMAGE_DIM = 512

# Final unified dimension for FAISS (we choose the larger = 512)
EMBED_DIM = max(TEXT_DIM, IMAGE_DIM)  

def _pad_vector(vec: np.ndarray) -> np.ndarray:
    """Pad vector to EMBED_DIM (float32)."""
    vec = np.asarray(vec, dtype="float32").reshape(-1)
    if vec.shape[0] == EMBED_DIM:
        return vec
    if vec.shape[0] > EMBED_DIM:
        return vec[:EMBED_DIM]
    padded = np.zeros((EMBED_DIM,), dtype="float32")
    padded[: vec.shape[0]] = vec
    return padded


def _get_index_paths(patient_id: str):
    index_path = os.path.join(INDEX_DIR, f"{patient_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{patient_id}_meta.json")
    return index_path, meta_path


def _load_or_create_index(patient_id: str):
    index_path, meta_path = _get_index_paths(patient_id)

    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(index_path)
        except Exception:
            # If read fails, recreate clean index
            index = faiss.IndexFlatL2(EMBED_DIM)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)
        metadata = []

    return index, metadata


def _save_index_and_meta(patient_id: str, index, metadata: List[Dict]):
    index_path, meta_path = _get_index_paths(patient_id)
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _chunk_text(text: str, max_chars=800, overlap=100) -> List[str]:
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    step = max_chars - overlap
    if step <= 0:
        step = max_chars

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start += step

    return chunks


# ADD TEXT
def add_text_to_index(patient_id: str, report_id: int, text: str):
    """
    Chunk text, compute text embeddings (384D) -> pad to 512D -> add to FAISS.
    """
    chunks = _chunk_text(text)
    if not chunks:
        return

    index, metadata = _load_or_create_index(patient_id)

    embeddings = TEXT_MODEL.encode(chunks, convert_to_numpy=True).astype("float32")
    padded_embeddings = np.vstack([_pad_vector(e) for e in embeddings])  # shape (n, EMBED_DIM)

    # ensure dim matches index
    if index.d != EMBED_DIM:
        # recreate index to correct dim
        index = faiss.IndexFlatL2(EMBED_DIM)

    index.add(padded_embeddings)

    base = len(metadata)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": base + i,
            "report_id": report_id,
            "type": "text",
            "text": chunk
        })

    _save_index_and_meta(patient_id, index, metadata)


# ADD IMAGE
def add_image_to_index(patient_id: str, report_id: int, embedding: np.ndarray):
    """
    embedding: raw image vector (likely 512D). Pad/truncate then add.
    """
    if embedding is None:
        return

    index, metadata = _load_or_create_index(patient_id)

    vec = _pad_vector(embedding).astype("float32").reshape(1, -1)

    if index.d != EMBED_DIM:
        index = faiss.IndexFlatL2(EMBED_DIM)

    index.add(vec)

    metadata.append({
        "chunk_id": len(metadata),
        "report_id": report_id,
        "type": "image",
        "text": "(IMAGE EMBEDDING)"
    })

    _save_index_and_meta(patient_id, index, metadata)


# SEARCH BY QUERY (text)
def search_patient_index(patient_id: str, query: str, top_k: int = 5) -> List[Dict]:
    index, metadata = _load_or_create_index(patient_id)
    if index.ntotal == 0 or not metadata:
        return []

    qvec = TEXT_MODEL.encode([query], convert_to_numpy=True)[0]
    qvec = _pad_vector(qvec).reshape(1, -1).astype("float32")

    distances, indices = index.search(qvec, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])

    return results


# SEARCH BY VECTOR (image -> nearest text chunks)
def search_by_vector(patient_id: str, vector: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Use a vector (image or text) to search the patient's index and return the top-k metadata entries.
    """
    index, metadata = _load_or_create_index(patient_id)
    if index.ntotal == 0 or not metadata:
        return []

    q = _pad_vector(vector).reshape(1, -1).astype("float32")
    distances, indices = index.search(q, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            # include distance for debugging usefulness
            entry = dict(metadata[idx])
            # add approximate similarity score
            results.append({"distance": float(distances[0][list(indices[0]).index(idx)]), **entry})

    return results

# UTILS
def get_patient_index_stats(patient_id: str):
    index, metadata = _load_or_create_index(patient_id)
    return {
        "patient_id": patient_id,
        "ntotal_vectors": int(index.ntotal),
        "metadata_len": len(metadata),
    }
