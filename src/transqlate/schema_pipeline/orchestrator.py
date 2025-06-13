# src/schema_pipeline/orchestrator.py

"""
orchestrator.py – RAG with dynamic threshold & safe fallback
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

from transqlate.schema_pipeline.formatter import format_schema, SPECIAL_TOKENS
from transqlate.schema_pipeline.graph import build_schema_graph
from transqlate.schema_pipeline.selector import build_table_embeddings, select_tables
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transqlate.embedding_utils import (
    EmbeddingDownloadError,
    load_sentence_embedder,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SchemaRAGOrchestrator:
    """
    End-to-end prompt builder:

    • Dynamically prunes schema if it exceeds token limits
    • Falls back to full schema if no tables pass the RAG threshold
    """

    def __init__(
        self,
        tokenizer,
        schema: Dict,
        embed_model: SentenceTransformer | None = None,
        base_sim_threshold: float = 0.20,
        max_schema_tokens: int = 1528,
        max_question_tokens: int = 64,
    ):
        self.tok = tokenizer
        self.base_sim_threshold = base_sim_threshold
        self.max_schema_tokens = max_schema_tokens
        self.max_question_tokens = max_question_tokens

        # Cache the full schema text & its token IDs
        self._schema = schema
        self._schema_text = format_schema(schema)
        self._schema_tokens: List[int] = self.tok.encode(
            self._schema_text, add_special_tokens=False
        )

        self._G = build_schema_graph(schema)
        self._embed = embed_model or load_sentence_embedder("all-MiniLM-L6-v2")
        self._table_embs = build_table_embeddings(schema, self._embed)

    def _encode_len(self, txt: str) -> int:
        try:
            return len(self.tok.encode(txt, add_special_tokens=False))
        except Exception:
            return len(txt.split()) * 2

    def _prune_schema(self, keep: List[str]) -> Tuple[str, List[int]]:
        ks = set(keep)
        pruned = {
            "tables": [t for t in self._schema["tables"] if t["name"] in ks],
            "foreign_keys": [
                fk
                for fk in self._schema.get("foreign_keys", [])
                if fk["from_table"] in ks and fk["to_table"] in ks
            ],
        }
        text = format_schema(pruned)
        return text, self.tok.encode(text, add_special_tokens=False)

    def build_prompt(self, question: str) -> Tuple[str, List[int], Dict]:
        q_len = self._encode_len(question)
        s_len = len(self._schema_tokens)
        use_rag = s_len > self.max_schema_tokens or q_len > self.max_question_tokens

        if use_rag:
            keep = select_tables(
                question,
                self._G,
                self._table_embs,
                self._embed,
                base_threshold=self.base_sim_threshold,
            )
            if keep:
                schema_text, _ = self._prune_schema(keep)
                rag_used = True
            else:
                logger.warning("No tables selected → reverting to full schema")
                schema_text, rag_used = self._schema_text, False
        else:
            schema_text, rag_used = self._schema_text, False

        prompt_txt = (
            f"{SPECIAL_TOKENS['NL_START']} {question} {SPECIAL_TOKENS['NL_END']} "
            f"{schema_text}"
        )
        prompt_ids = self.tok.encode(prompt_txt, add_special_tokens=False)

        info = {"rag_used": rag_used, "prompt_tokens": len(prompt_ids)}
        logger.debug("Prompt built | rag=%s | tokens=%d", rag_used, len(prompt_ids))
        return prompt_txt, prompt_ids, info


# ── Optional smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, pathlib

    root = pathlib.Path(__file__).parent
    try:
        schema = json.loads((root / "sample_schema.json").read_text())
        # Ensure forward slashes for HuggingFace repo IDs
        def hf_model_id(model_id):
            return model_id.replace("\\", "/")
        tok = AutoTokenizer.from_pretrained(hf_model_id("Shaurya-Sethi/transqlate-phi4"), use_fast=True)
        orch = SchemaRAGOrchestrator(tok, schema)
        ptxt, pids, meta = orch.build_prompt(
            "Which countries saw a >10% population increase in 2020?"
        )
        print(meta, "\n", ptxt[:140], "…")
    except FileNotFoundError:
        print("Smoke-test skipped (no sample_schema.json found)")
