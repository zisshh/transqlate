"""Utility helpers for sentence embedding models."""

from __future__ import annotations

import threading
from typing import Optional

from rich.console import Console

try:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional heavy deps
    SentenceTransformer = None  # type: ignore
    snapshot_download = None  # type: ignore

console = Console()


class EmbeddingDownloadError(RuntimeError):
    """Raised when the embedding model could not be downloaded."""


def _is_cached(model_id: str) -> bool:
    if snapshot_download is None:
        return False
    try:
        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def load_sentence_embedder(model_id: str = "all-MiniLM-L6-v2", timeout: int = 120) -> SentenceTransformer:
    """Load a sentence-transformer model with a spinner and timeout."""
    if SentenceTransformer is None:
        raise EmbeddingDownloadError("sentence-transformers library is not installed")

    cached = _is_cached(model_id)
    msg = "Loading sentence embedding model from Hugging Face..."
    if cached:
        msg = "Loading sentence embedding model from cache..."

    model: Optional[SentenceTransformer] = None
    err: Optional[Exception] = None

    def _load():
        nonlocal model, err
        try:
            model = SentenceTransformer(model_id)
        except Exception as e:  # pragma: no cover - network errors
            err = e

    thread = threading.Thread(target=_load)
    with console.status(f"[bold cyan]{msg}[/bold cyan]", spinner="dots"):
        thread.start()
        thread.join(timeout)
    if thread.is_alive():
        err = TimeoutError(f"Download timed out after {timeout} seconds")
        thread.join()

    if err or model is None:
        raise EmbeddingDownloadError(str(err) if err else "Unknown error")
    return model

