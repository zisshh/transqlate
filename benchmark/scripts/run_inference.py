#!/usr/bin/env python
"""Run Transqlate inference on the SPIDER test set."""

# Dependencies: json, pathlib, logging, torch, tqdm, transformers, sentence-transformers,
# and local transqlate modules.

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer  # noqa: F401 - document dependency
from sentence_transformers import SentenceTransformer  # noqa: F401 - document dependency

from transqlate.inference import NL2SQLInference
from transqlate.schema_pipeline.orchestrator import SchemaRAGOrchestrator
from transqlate.embedding_utils import load_sentence_embedder


logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Transqlate inference on the SPIDER test set."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Index of the first example to process",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Index after the last example to process",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N examples",
    )
    return parser.parse_args(argv)


def save_checkpoint(path: Path, predictions: list, index: int) -> None:
    """Write checkpoint data to ``path``."""
    try:
        data = {"index": index, "predictions": predictions}
        path.write_text(json.dumps(data))
        logger.info("Checkpoint saved at index %s to %s", index, path)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Failed to save checkpoint: %s", exc)


def load_checkpoint(path: Path) -> tuple[list, int]:
    """Return predictions and next index from ``path``."""
    try:
        data = json.loads(path.read_text())
        predictions = data.get("predictions", [])
        index = int(data.get("index", len(predictions)))
        return predictions, index
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Failed to load checkpoint %s: %s", path, exc)
        return [], 0


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


def main(argv=None) -> None:
    """Entry point for the benchmark script."""
    args = parse_args(argv)
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    test_path = data_dir / "test.json"
    tables_path = data_dir / "test_tables.json"

    logger.info("Loading data from %s", data_dir)
    examples = json.loads(test_path.read_text())
    table_entries = json.loads(tables_path.read_text())
    schemas = {entry["db_id"]: transform_schema(entry) for entry in table_entries}

    cp_path = args.checkpoint or results_dir / "predictions_checkpoint.json"
    predictions: list[str] = []
    start_index = 0
    if cp_path.exists():
        predictions, start_index = load_checkpoint(cp_path)

    if args.start_index is not None:
        start_index = args.start_index

    end_index = args.end_index if args.end_index is not None else len(examples)
    predictions = predictions[:start_index]
    failed: list[int] = []

    inference = NL2SQLInference()
    embed_model = load_sentence_embedder("all-MiniLM-L6-v2")
    orchestrators: dict[str, SchemaRAGOrchestrator] = {}

    current_index = start_index

    def handle_signal(signum, frame):  # noqa: D401 - callback signature
        del frame
        logger.warning("Received signal %s; saving checkpoint", signum)
        save_checkpoint(cp_path, predictions, current_index)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.time()
    with tqdm(
        range(start_index, end_index),
        desc="Infer",
        unit="ex",
        initial=start_index,
        total=end_index,
    ) as progress:
        for idx in progress:
            current_index = idx
            ex = examples[idx]
            question = ex.get("question", "")
            db_id = ex.get("db_id")
            try:
                schema = schemas[db_id]
                orchestrator = orchestrators.get(db_id)
                if orchestrator is None:
                    orchestrator = SchemaRAGOrchestrator(
                        inference.tokenizer, schema, embed_model=embed_model
                    )
                    orchestrators[db_id] = orchestrator
                prompt_text, _, _ = orchestrator.build_prompt(question)
                schema_tokens = extract_schema_token_span(
                    prompt_text, inference.tokenizer
                )
                _, sql_text = inference.generate(question, schema_tokens)
                predictions.append(sql_text.strip())
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.exception(
                    "Failed example %s (db_id=%s): %s", idx, db_id, exc
                )
                predictions.append("")
                failed.append(idx)
            if (idx + 1 - start_index) % args.checkpoint_interval == 0:
                save_checkpoint(cp_path, predictions, idx + 1)

    elapsed = time.time() - start_time
    save_checkpoint(cp_path, predictions, end_index)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "predictions.sql"
    with out_path.open("w", encoding="utf-8") as fh:
        for line in predictions:
            fh.write(line + "\n")

    if failed:
        fail_path = results_dir / "failed_examples.txt"
        fail_path.write_text("\n".join(str(i) for i in failed))
        logger.info("Failed indices written to %s", fail_path)

    processed = end_index - start_index
    speed = processed / elapsed if elapsed else 0.0
    logger.info(
        "Processed %s examples with %s failures in %.1fs (%.2f ex/s)",
        processed,
        len(failed),
        elapsed,
        speed,
    )
    logger.info("Predictions written to %s", out_path)


if __name__ == "__main__":
    main()
