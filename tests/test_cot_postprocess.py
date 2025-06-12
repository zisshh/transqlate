import sys
import types
import importlib
from pathlib import Path

# Helper to import cli module with stubbed heavy dependencies

def load_fix_cot_dbms():
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
    # minimal attributes to satisfy imports
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
    return cli._fix_cot_dbms


_fix_cot_dbms = load_fix_cot_dbms()


def test_fix_cot_dbms_postgres():
    text = "To translate the natural language question into an executable SQLite query, we need to follow these steps:"
    expected = "To translate the natural language question into an executable PostgreSQL query, we need to follow these steps:"
    assert _fix_cot_dbms(text, "postgres") == expected


def test_fix_cot_dbms_case_insensitive():
    text = "something about SQLITE and mysql"
    result = _fix_cot_dbms(text, "oracle")
    assert "Oracle" in result
    assert "SQLITE" not in result and "mysql" not in result
