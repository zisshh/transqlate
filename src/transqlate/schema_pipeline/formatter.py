# src/schema_pipeline/formatter.py

"""
formatter.py – Schema → single-line text for NL2SQL
"""

from __future__ import annotations
from typing import Dict, List

# Must match the tokens used during fine-tuning
SPECIAL_TOKENS = {
    "SCHEMA_START": "<SCHEMA>",
    "SCHEMA_END": "</SCHEMA>",
    "PK_START": "<PK>",
    "PK_END": "</PK>",
    "FK_START": "<FK>",
    "FK_END": "</FK>",
    "NL_START": "<NL>",
    "NL_END": "</NL>",
    "COT_START": "<COT>",
    "COT_END": "</COT>",
    "SQL_START": "<SQL>",
    "SQL_END": "</SQL>",
    "EXT_START": "<EXT>",
    "EXT_END": "</EXT>",
}


def _format_table(table: Dict, fk_map: Dict[str, set]) -> str:
    """Return single-table string with each PK/FK wrapped appropriately."""
    tname = table["name"]
    pk_cols = [c for c in table["columns"] if c.get("pk")]
    fk_cols = fk_map.get(tname, set())

    chunks: List[str] = []
    # 1) Primary keys
    for c in pk_cols:
        chunks.append(
            f"{SPECIAL_TOKENS['PK_START']} {c['name']}:{c['type']} {SPECIAL_TOKENS['PK_END']}"
        )
    # 2) Other columns
    for col in table["columns"]:
        if col.get("pk"):
            continue
        name, typ = col["name"], col["type"]
        if name in fk_cols:
            chunks.append(
                f"{SPECIAL_TOKENS['FK_START']} {name}:{typ} {SPECIAL_TOKENS['FK_END']}"
            )
        else:
            chunks.append(f"{name}:{typ}")

    return f"{tname}({', '.join(chunks)})"


def format_schema(schema: Dict) -> str:
    """
    Convert the extractor schema dict into the single-line form:
      <SCHEMA> table1(...) table2(...) ... </SCHEMA>
    """
    # Build lookup for foreign keys
    fk_map: Dict[str, set] = {}
    for fk in schema.get("foreign_keys", []):
        fk_map.setdefault(fk["from_table"], set()).add(fk["from_column"])

    tables_txt = " ".join(_format_table(t, fk_map) for t in schema["tables"])
    return f"{SPECIAL_TOKENS['SCHEMA_START']} {tables_txt} {SPECIAL_TOKENS['SCHEMA_END']}"
