#!/usr/bin/env python
"""Run Transqlate inference on the SPIDER test set."""

# Dependencies: json, pathlib, logging, torch, tqdm, transformers, sentence-transformers,
# and local transqlate modules.

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer  # noqa: F401 - document dependency
from sentence_transformers import SentenceTransformer  # noqa: F401 - document dependency

from transqlate.inference import NL2SQLInference
from transqlate.schema_pipeline.orchestrator import SchemaRAGOrchestrator
from transqlate.embedding_utils import load_sentence_embedder


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def transform_schema(spider_schema: dict) -> dict:
    """Convert SPIDER schema format to SchemaRAGOrchestrator format."""
    table_names = spider_schema.get("table_names_original") or spider_schema["table_names"]
    column_names = spider_schema.get("column_names_original") or spider_schema["column_names"]
    column_types = spider_schema["column_types"]
    pk_set = set(spider_schema.get("primary_keys", []))

    tables = [{"name": name, "columns": []} for name in table_names]
    col_index_map = {}

    for idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx == -1:
            continue
        col_info = {"name": col_name, "type": column_types[idx]}
        if idx in pk_set:
            col_info["pk"] = True
        tables[tbl_idx]["columns"].append(col_info)
        col_index_map[idx] = (table_names[tbl_idx], col_name)

    foreign_keys = []
    for from_idx, to_idx in spider_schema.get("foreign_keys", []):
        from_info = col_index_map.get(from_idx)
        to_info = col_index_map.get(to_idx)
        if not from_info or not to_info:
            continue
        foreign_keys.append(
            {
                "from_table": from_info[0],
                "from_column": from_info[1],
                "to_table": to_info[0],
                "to_column": to_info[1],
            }
        )

    return {"tables": tables, "foreign_keys": foreign_keys}


def extract_schema_token_span(prompt_text: str, tokenizer):
    """Return token span covering <SCHEMA> ... </SCHEMA> in the encoded prompt."""
    schema_start_str = "<SCHEMA>"
    schema_end_str = "</SCHEMA>"
    try:
        sch_start_char = prompt_text.index(schema_start_str)
        sch_end_char = prompt_text.index(schema_end_str) + len(schema_end_str)
    except ValueError as exc:
        raise ValueError("Could not find <SCHEMA> or </SCHEMA> in the prompt text.") from exc
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    sch_start_token_idx = None
    sch_end_token_idx = None
    for i, (start, end) in enumerate(offsets):
        if sch_start_token_idx is None and start <= sch_start_char < end:
            sch_start_token_idx = i
        if sch_end_token_idx is None and start < sch_end_char <= end:
            sch_end_token_idx = i
            break
    if sch_end_token_idx is None:
        for i, (start, end) in reversed(list(enumerate(offsets))):
            if start < sch_end_char:
                sch_end_token_idx = i
                break
    if sch_start_token_idx is None or sch_end_token_idx is None:
        raise ValueError(
            f"Could not map <SCHEMA> or </SCHEMA> to token positions. sch_start_token_idx: {sch_start_token_idx}, sch_end_token_idx: {sch_end_token_idx}, offsets: {offsets}"
        )
    schema_tokens = enc["input_ids"][sch_start_token_idx : sch_end_token_idx + 1]
    return schema_tokens


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    test_path = data_dir / "test.json"
    tables_path = data_dir / "test_tables.json"

    logger.info("Loading data from %s", data_dir)
    examples = json.loads(test_path.read_text())
    table_entries = json.loads(tables_path.read_text())
    schemas = {entry["db_id"]: transform_schema(entry) for entry in table_entries}

    inference = NL2SQLInference()
    embed_model = load_sentence_embedder("all-MiniLM-L6-v2")
    orchestrators = {}
    predictions = []

    for ex in tqdm(examples, desc="Infer", unit="ex"):
        question = ex.get("question", "")
        db_id = ex.get("db_id")
        schema = schemas.get(db_id)
        if schema is None:
            logger.error("Schema for db_id %s not found", db_id)
            predictions.append("")
            continue
        orchestrator = orchestrators.get(db_id)
        if orchestrator is None:
            orchestrator = SchemaRAGOrchestrator(
                inference.tokenizer, schema, embed_model=embed_model
            )
            orchestrators[db_id] = orchestrator
        prompt_text, _, _ = orchestrator.build_prompt(question)
        try:
            schema_tokens = extract_schema_token_span(prompt_text, inference.tokenizer)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error("Failed to locate schema tokens for %s: %s", db_id, exc)
            predictions.append("")
            continue
        _, sql_text = inference.generate(question, schema_tokens)
        predictions.append(sql_text.strip())

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "predictions.sql"
    with out_path.open("w", encoding="utf-8") as fh:
        for line in predictions:
            fh.write(line + "\n")
    logger.info("Predictions written to %s", out_path)


if __name__ == "__main__":
    main()
