import sys
import types
import importlib
from pathlib import Path


def load_session():
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
    return cli.Session


Session = load_session()


class DummyCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self.rowcount = 0

    def execute(self, sql):
        self.conn.last_sql = sql

    def fetchall(self):
        return []


class DummyConnection:
    def __init__(self):
        self.last_sql = None

    def cursor(self):
        return DummyCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


class DummyExtractor:
    def __init__(self):
        self.conn = DummyConnection()
        self.default_schema = ""
        self.user = ""


def test_oracle_semicolon_removed():
    sess = Session(
        "oracle",
        DummyExtractor(),
        {"tables": []},
        None,
        None,
        None,
    )
    sess.table_schemas = {}
    sess.execute_sql("SELECT * FROM MY_MEMBERS;")
    assert sess.extractor.conn.last_sql == "SELECT * FROM MY_MEMBERS"
