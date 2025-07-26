import json
from pathlib import Path
import sys

import pytest


@pytest.fixture()
def setup_data(tmp_path):
    bench_dir = tmp_path / "benchmark"
    data_dir = bench_dir / "data"
    scripts_dir = bench_dir / "scripts"
    results_dir = bench_dir / "results"
    data_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    results_dir.mkdir()

    examples = [
        {"question": "q1", "db_id": "db1"},
        {"question": "q2", "db_id": "db1"},
    ]
    (data_dir / "test.json").write_text(json.dumps(examples))
    (data_dir / "test_tables.json").write_text(json.dumps([{"db_id": "db1"}]))
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

    return bench_dir, scripts_dir


def test_embedder_loaded_once(monkeypatch, setup_data):
    bench_dir, scripts_dir = setup_data
    import importlib.util
    import sys
    import types
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    tfm_mod = types.ModuleType("transformers")
    tfm_mod.AutoTokenizer = object
    sys.modules.setdefault("transformers", tfm_mod)
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", st_mod)
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda *a, **k: a[0] if a else []
    sys.modules.setdefault("tqdm", tqdm_mod)
    # stub transqlate modules to avoid heavy imports during load
    transqlate_mod = types.ModuleType("transqlate")
    sys.modules.setdefault("transqlate", transqlate_mod)
    inf_mod = types.ModuleType("transqlate.inference")
    inf_mod.NL2SQLInference = object
    sys.modules.setdefault("transqlate.inference", inf_mod)
    schema_pkg = types.ModuleType("transqlate.schema_pipeline")
    sys.modules.setdefault("transqlate.schema_pipeline", schema_pkg)
    orch_mod = types.ModuleType("transqlate.schema_pipeline.orchestrator")
    orch_mod.SchemaRAGOrchestrator = object
    sys.modules.setdefault("transqlate.schema_pipeline.orchestrator", orch_mod)
    emb_mod = types.ModuleType("transqlate.embedding_utils")
    emb_mod.load_sentence_embedder = lambda *a, **k: None
    sys.modules.setdefault("transqlate.embedding_utils", emb_mod)
    spec = importlib.util.spec_from_file_location(
        "run_inference",
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "scripts"
        / "run_inference.py",
    )
    run_inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_inference)  # type: ignore

    monkeypatch.setattr(run_inference, "__file__", str(scripts_dir / "run_inference.py"))

    class DummyInf:
        def __init__(self):
            self.tokenizer = object()

        def generate(self, question, tokens):
            return None, "sql"

    monkeypatch.setattr(run_inference, "NL2SQLInference", DummyInf)

    counts = {"embed": 0, "orch": 0}

    def fake_load_sentence_embedder(model_id="all-MiniLM-L6-v2"):
        counts["embed"] += 1
        return "embed"

    monkeypatch.setattr(run_inference, "load_sentence_embedder", fake_load_sentence_embedder)

    class DummyOrchestrator:
        def __init__(self, tok, schema, embed_model=None):
            assert embed_model == "embed"
            counts["orch"] += 1

        def build_prompt(self, question):
            return "prompt", [], {}

    monkeypatch.setattr(run_inference, "SchemaRAGOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(run_inference, "extract_schema_token_span", lambda *a, **k: [])
    monkeypatch.setattr(run_inference, "transform_schema", lambda e: {})

    run_inference.main()

    assert counts["embed"] == 1
    assert counts["orch"] == 1
