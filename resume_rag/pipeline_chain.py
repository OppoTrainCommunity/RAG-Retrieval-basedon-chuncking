# resume_rag/pipeline_chain.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import hashlib

from .parse_pdf import extract_text_from_pdf
from .chunker import chunk_text
from .vector_store_chain import add_texts
from .config import AppConfig

def _make_resume_id(filename: str, text: str) -> str:
    h = hashlib.sha1((filename + "\n" + text[:2000]).encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]

def _index_path(cfg: AppConfig) -> Path:
    return Path(cfg.persist_dir) / cfg.index_file

def load_resume_index(cfg: AppConfig) -> Dict[str, Any]:
    p = _index_path(cfg)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_resume_index(cfg: AppConfig, data: Dict[str, Any]) -> None:
    p = _index_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def list_resumes(cfg: AppConfig) -> List[Dict[str, Any]]:
    idx = load_resume_index(cfg)
    items = []
    for rid, meta in idx.items():
        items.append({"resume_id": rid, **meta})
    # newest first if timestamp exists
    items.sort(key=lambda x: x.get("indexed_at", 0), reverse=True)
    return items

def index_resume_pdf(
    cfg: AppConfig,
    pdf_path: str,
    original_filename: str,
    mode: str,
    chunk_size: int,
    overlap: int,
) -> Dict[str, Any]:
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No text extracted from PDF (maybe scanned image).")

    resume_id = _make_resume_id(original_filename, text)
    chunks = chunk_text(text, mode=mode, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    # metadata per chunk
    metadatas = []
    ids = []
    for i, ch in enumerate(chunks):
        ids.append(f"{resume_id}-{i}")
        metadatas.append({
            "resume_id": resume_id,
            "filename": original_filename,
            "chunk_id": i,
            "chunk_mode": mode,
        })

    add_texts(
        collection_name=cfg.collection_name,
        persist_dir=cfg.persist_dir,
        texts=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    # update index file
    import time
    idx = load_resume_index(cfg)
    idx[resume_id] = {
        "filename": original_filename,
        "num_chunks": len(chunks),
        "chunk_mode": mode,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "indexed_at": int(time.time()),
    }
    save_resume_index(cfg, idx)

    return {
        "resume_id": resume_id,
        "filename": original_filename,
        "num_chunks": len(chunks),
        "text_preview": text[:1200],
    }

def reset_index(cfg: AppConfig) -> None:
    p = _index_path(cfg)
    if p.exists():
        p.unlink()
