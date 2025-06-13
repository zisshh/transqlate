import sys
import types
import importlib
from pathlib import Path


def load_post_process():
    stub_names = {
        "rich": types.ModuleType("rich"),
        "rich.console": types.ModuleType("rich.console"),
        "rich.panel": types.ModuleType("rich.panel"),
        "rich.prompt": types.ModuleType("rich.prompt"),
        "rich.table": types.ModuleType("rich.table"),
        "transqlate.schema_pipeline.extractor": types.ModuleType("dummy"),
        "transqlate.schema_pipeline.formatter": types.ModuleType("dummy"),
        "transqlate.schema_pipeline.orchestrator": types.ModuleType("dummy"),
        "transqlate.schema_pipeline.selector": types.ModuleType("dummy"),
        "transqlate.inference": types.ModuleType("dummy"),
        "transformers": types.ModuleType("dummy"),
    }
    stub_names["rich.console"].Console = object
    stub_names["rich.panel"].Panel = object
    stub_names["rich.prompt"].Prompt = object
    stub_names["rich.table"].Table = object
    stub_names["transqlate.schema_pipeline.extractor"].get_schema_extractor = lambda *a, **k: None
    stub_names["transqlate.schema_pipeline.formatter"].format_schema = lambda *a, **k: ""
    stub_names["transqlate.schema_pipeline.orchestrator"].SchemaRAGOrchestrator = object
    stub_names["transqlate.schema_pipeline.selector"].build_table_embeddings = lambda *a, **k: None
    stub_names["transqlate.inference"].NL2SQLInference = object
    stub_names["transformers"].AutoTokenizer = object

    saved = {name: sys.modules.get(name) for name in stub_names}
    sys.modules.update(stub_names)
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    try:
        cli = importlib.import_module("transqlate.cli.cli")
    finally:
        for name, mod in saved.items():
            if mod is None:
                del sys.modules[name]
            else:
                sys.modules[name] = mod
    return cli._post_process_mssql_sql


_post_process = load_post_process()


def test_limit_to_top_and_quotes():
    sql = 'SELECT "name" FROM users ORDER BY id LIMIT 5;'
    expected = 'SELECT TOP 5 [name] FROM users ORDER BY id;'
    assert _post_process(sql) == expected


def test_boolean_literals():
    sql = 'SELECT * FROM t WHERE active=TRUE AND deleted=FALSE;'
    expected = 'SELECT * FROM t WHERE active=1 AND deleted=0;'
    assert _post_process(sql) == expected


def test_autoincrement_and_datatypes():
    sql = 'CREATE TABLE t (`id` INTEGER PRIMARY KEY AUTOINCREMENT, created TIMESTAMP);'
    expected = 'CREATE TABLE t ([id] INT PRIMARY KEY IDENTITY(1,1), created DATETIME2);'
    assert _post_process(sql) == expected


def test_existing_top_preserved():
    sql = 'SELECT TOP 5 name FROM users LIMIT 10;'
    expected = 'SELECT TOP 5 name FROM users;'
    assert _post_process(sql) == expected


def test_backticks_and_no_semicolon():
    sql = 'SELECT `Name` FROM "Users" LIMIT 1'
    expected = 'SELECT TOP 1 [Name] FROM [Users]'
    assert _post_process(sql) == expected
