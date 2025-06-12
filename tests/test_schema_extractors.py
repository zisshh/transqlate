import sys
from pathlib import Path
from unittest import mock
import sqlite3
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from transqlate.schema_pipeline.extractor import (
    get_schema_extractor,
    SQLiteSchemaExtractor,
    PostgresSchemaExtractor,
    MySQLSchemaExtractor,
    MSSQLSchemaExtractor,
    OracleSchemaExtractor,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "simple.db"
    conn = sqlite3.connect(path)
    with open(FIXTURES / "schema.sql", "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return str(path)

EXPECTED = {
    "tables": [
        {
            "name": "users",
            "columns": [
                {"name": "id", "type": "INTEGER", "pk": True},
                {"name": "name", "type": "TEXT", "pk": False},
            ],
        },
        {
            "name": "posts",
            "columns": [
                {"name": "id", "type": "INTEGER", "pk": True},
                {"name": "user_id", "type": "INTEGER", "pk": False},
                {"name": "content", "type": "TEXT", "pk": False},
            ],
        },
    ],
    "foreign_keys": [
        {
            "from_table": "posts",
            "from_column": "user_id",
            "to_table": "users",
            "to_column": "id",
        }
    ],
}


def _check_schema(schema):
    exp = {"tables": [], "foreign_keys": list(EXPECTED["foreign_keys"]) }
    for tbl in EXPECTED["tables"]:
        exp["tables"].append({"name": tbl["name"], "columns": list(tbl["columns"])})

    def sort(s):
        s["tables"].sort(key=lambda t: t["name"])
        for t in s["tables"]:
            t["columns"].sort(key=lambda c: c["name"])
        s["foreign_keys"].sort(key=lambda fk: (fk["from_table"], fk["from_column"]))

    sort(exp)
    sort(schema)
    assert schema == exp


def test_sqlite_extractor(db_path):
    ex = get_schema_extractor("sqlite", db_path=db_path)
    schema = ex.extract_schema()
    _check_schema(schema)


def _make_dummy(dbms_name: str, path: str):
    class Dummy(SQLiteSchemaExtractor):
        dbms = dbms_name

        def __init__(self, *_, **__):
            super().__init__(path)

    return Dummy

def test_all_other_extractors(db_path):
    patches = [
        mock.patch("transqlate.schema_pipeline.extractor.PostgresSchemaExtractor", _make_dummy("postgresql", db_path)),
        mock.patch("transqlate.schema_pipeline.extractor.MySQLSchemaExtractor", _make_dummy("mysql", db_path)),
        mock.patch("transqlate.schema_pipeline.extractor.MSSQLSchemaExtractor", _make_dummy("mssql", db_path)),
        mock.patch("transqlate.schema_pipeline.extractor.OracleSchemaExtractor", _make_dummy("oracle", db_path)),
    ]
    for p in patches:
        p.start()
    try:
        db_types = ["postgresql", "mysql", "mssql", "oracle"]
        for db in db_types:
            ex = get_schema_extractor(db, host="", user="", password="", dbname="", database="", service_name="")
            schema = ex.extract_schema()
            assert ex.dbms == db
            _check_schema(schema)
    finally:
        for p in patches:
            p.stop()

