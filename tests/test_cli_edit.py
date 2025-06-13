import sys
import types
import importlib
from pathlib import Path


def load_prompt_functions():
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
    class DummyConsole:
        def print(self, *a, **k):
            pass
    stub_names["rich.console"].Console = DummyConsole
    stub_names["rich.panel"].Panel = lambda *a, **k: None
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
    return cli._prompt_edit_sql, cli._prompt_write_sql

_prompt_edit_sql, _prompt_write_sql = load_prompt_functions()


def test_finish(monkeypatch):
    inputs = iter(["SELECT 1;", ":finish"])
    def fake_input(prompt=""):
        return next(inputs)
    monkeypatch.setattr("builtins.input", fake_input)
    result = _prompt_edit_sql("SELECT 0;")
    assert result == "SELECT 1;"


def test_cancel(monkeypatch):
    inputs = iter([":cancel"])
    def fake_input(prompt=""):
        return next(inputs)
    monkeypatch.setattr("builtins.input", fake_input)
    result = _prompt_edit_sql("SELECT 0;")
    assert result is None


def test_multiline(monkeypatch):
    inputs = iter(["SELECT", "*", "FROM t", ":finish"])
    def fake_input(prompt=""):
        return next(inputs)
    monkeypatch.setattr("builtins.input", fake_input)
    result = _prompt_edit_sql("SELECT * FROM old")
    assert result == "SELECT\n*\nFROM t"


def test_write_cancel(monkeypatch):
    inputs = iter([":cancel"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    result = _prompt_write_sql()
    assert result is None


def test_write_multiline(monkeypatch):
    inputs = iter(["SELECT", "*", "FROM x", ":finish"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    result = _prompt_write_sql()
    assert result == "SELECT\n*\nFROM x"
