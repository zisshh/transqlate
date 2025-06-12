# src/transqlate/cli/cli.py
# --------------------------------------------------------------------------
# Transqlate CLI – powered by the fine-tuned Phi-4-mini (QLoRA) model on HF Hub
#  ▸ Loads tokenizer and model directly from Hugging Face or a local path
#  ▸ Output parsing keys off the first "SQL:" label (matches new dataset)
#  ▸ Now safely prompts before running any DDL/DML, and handles all result types
# --------------------------------------------------------------------------

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import re
import traceback
from getpass import getpass
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ─── Silence noisy libs ──────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

try:
    from sentence_transformers import logging as st_logging
    st_logging.set_verbosity_warning()
except ImportError:
    pass

# ─── Project imports ─────────────────────────────────────────────────────
from transqlate.schema_pipeline.extractor import get_schema_extractor
from transqlate.schema_pipeline.formatter import format_schema
from transqlate.schema_pipeline.orchestrator import SchemaRAGOrchestrator
from transqlate.schema_pipeline.selector import build_table_embeddings
from transqlate.inference import NL2SQLInference

from transformers import AutoTokenizer

console = Console()
_SQL_SPLIT_RE = re.compile(r"\bSQL\s*:\s*", re.IGNORECASE)

# Map supported DB types to display names used in the CLI
_DB_DISPLAY_NAMES = {
    "sqlite": "SQLite",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mssql": "MSSQL",
    "sqlserver": "MSSQL",
    "oracle": "Oracle",
}

_DB_NAME_PATTERN = re.compile(
    r"\b(sqlite|postgresql|postgres|mysql|mssql|sqlserver|oracle)\b",
    re.IGNORECASE,
)

# Toggle for showing Python tracebacks with errors. Can be enabled via
# the --tracebacks CLI flag during debugging.
SHOW_TRACEBACKS = False

def extract_sql(sql_text: str, cot_text: str) -> str:
    candidate = sql_text.strip()
    is_incomplete = (
        not candidate
        or not candidate.rstrip().endswith(";")
        or candidate.lower().count("select") > 1
        or candidate.lower().endswith("group by")
        or candidate.count("\n") < 2
    )
    if is_incomplete:
        match = re.search(r"```sql\s*(.+?)```", cot_text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
        else:
            match2 = re.findall(r"(SELECT|INSERT|UPDATE|DELETE).*?;", cot_text, re.IGNORECASE | re.DOTALL)
            if match2:
                candidate = match2[-1]
            else:
                candidate = ""
    candidate = re.sub(r"<\|endoftext\|>", "", candidate)
    candidate = "\n".join(line for line in candidate.splitlines() if not line.strip().startswith("--"))
    candidate = candidate.replace("```", "")
    return candidate.strip()

def _print_exception(exc: Exception):
    """Display a brief error message and, optionally, the traceback."""
    console.print(Panel.fit(f"[bold red]Error:[/bold red] {exc}", style="red"))
    if SHOW_TRACEBACKS:
        traceback.print_exc()


def _fix_cot_dbms(cot_text: str, db_type: str) -> str:
    """Replace any DBMS names in `cot_text` with the current database type."""
    desired = _DB_DISPLAY_NAMES.get(db_type.lower(), db_type.capitalize())

    def repl(match: re.Match) -> str:
        return desired

    return _DB_NAME_PATTERN.sub(repl, cot_text)

def _collect_db_params(db_type: str) -> Tuple[str, dict]:
    params = {}
    if db_type == "sqlite":
        while True:
            msg = (
                "SQLite database file path\n"
                "[yellow]Paste or type the full path to your .db file (e.g., C:\\Users\\me\\mydata.db).[/yellow]\n"
                "[yellow]Do NOT wrap the path in quotes. If you copied it with quotes, remove them.[/yellow]"
            )
            db_path = Prompt.ask(msg, default="example.db")
            if (db_path.startswith('"') and db_path.endswith('"')) or (db_path.startswith("'") and db_path.endswith("'")):
                console.print(
                    "[red]Please remove the wrapping quotes from your path and try again.[/red]"
                )
                continue
            params["db_path"] = db_path
            break
    else:
        params["host"] = Prompt.ask("Host", default="localhost")
        default_port = {
            "postgres": "5432",
            "postgresql": "5432",
            "mysql": "3306",
            "mssql": "1433",
            "oracle": "1521",
        }.get(db_type, "5432")
        params["port"] = int(Prompt.ask("Port", default=default_port))
        if db_type in {"postgres", "postgresql"}:
            params["dbname"] = Prompt.ask("Database name")
        else:
            params["database"] = Prompt.ask("Database name")
        params["user"] = Prompt.ask("Username")
        params["password"] = getpass("Password: ")
        if db_type == "oracle":
            params["service_name"] = params.pop("database")
    return db_type, params

def _choose_db_interactively() -> Tuple[str, dict]:
    console.print(
        Panel(
            "[bold cyan]Let's connect to your database![/bold cyan]\n"
            "You'll be asked for connection details based on your database type.\n"
            "[green]Tip:[/green] For SQLite, you just need the path to your .db file.",
            style="cyan",
        )
    )
    db_type = Prompt.ask(
        "Choose DB type",
        choices=["sqlite", "postgres", "mysql", "mssql", "oracle"],
        default="sqlite",
    )
    return _collect_db_params(db_type)

class Session:
    DDL_DML_PATTERN = re.compile(
        r"^\s*(DROP|CREATE|ALTER|TRUNCATE|RENAME|INSERT|UPDATE|DELETE)\b", re.IGNORECASE
    )

    def __init__(
        self,
        db_type: str,
        extractor,
        schema_dict: dict,
        tokenizer,
        orchestrator: SchemaRAGOrchestrator,
        inference: NL2SQLInference,
        table_embs=None,
    ):
        self.db_type = db_type
        self.extractor = extractor
        self.schema_dict = schema_dict
        self.tokenizer = tokenizer
        self.orchestrator = orchestrator
        self.inference = inference
        self.history: List[Tuple[str, str]] = []
        self.table_embs = table_embs

    def execute_sql(self, sql: str):
        # -- DDL/DML confirmation prompt --
        if self.DDL_DML_PATTERN.match(sql):
            console.print(Panel(
                f"[red]Caution: This statement will alter your database.[/red]\n"
                f"[bold yellow]{sql}[/bold yellow]",
                style="red"
            ))
            resp = Prompt.ask("[bold red]Are you sure you want to execute this statement?[/bold red] (y/N)", default="N")
            if resp.strip().lower() not in ("y", "yes"):
                console.print("[yellow]Cancelled. Statement not executed.[/yellow]")
                return

        try:
            cur = self.extractor.conn.cursor()
            cur.execute(sql)
            if cur.description is None:
                affected = cur.rowcount
                try:
                    self.extractor.conn.commit()  # commit for DML/DDL (safely ignored for SELECT)
                except Exception:
                    pass
                msg = f"[green]Statement executed.[/green] [dim]{affected if affected >= 0 else ''} row(s) affected.[/dim]"
                console.print(msg)
            else:
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                self._pretty_table(rows, cols)
        except Exception as e:
            try:
                self.extractor.conn.rollback()
            except Exception:
                pass
            msg = str(e)
            lower = msg.lower()
            if "current transaction is aborted" in lower or "infailedsqltransaction" in lower:
                console.print(
                    Panel(
                        "A previous query failed and the transaction was aborted. The connection has been reset. Please retry your query.",
                        style="red",
                    )
                )
            elif re.search(r"no such table|does not exist|undefined table", lower):
                console.print(
                    Panel(
                        f"{msg}\n[dim]Use :show schema to view available tables.[/dim]",
                        style="yellow",
                    )
                )
            else:
                console.print(Panel(f"Error executing query:\n{msg}", style="red"))

    def _pretty_table(self, rows, cols):
        if not rows:
            console.print("[yellow]No rows returned.[/yellow]")
            return
        table = Table(show_header=True, header_style="bold magenta")
        for c in cols:
            table.add_column(str(c))
        for r in rows:
            table.add_row(*[str(cell) for cell in r])
        console.print(table)

    def suggest_schema_terms(self, token: str, top_k: int = 5) -> List[str]:
        try:
            from rapidfuzz import process as fuzz_process
        except ImportError:
            return []
        names = [t["name"] for t in self.schema_dict["tables"]]
        cols = [
            f"{t['name']}.{c['name']}"
            for t in self.schema_dict["tables"]
            for c in t["columns"]
        ]
        candidates = names + cols
        return [
            s for s, _ in fuzz_process.extract(token, candidates, limit=top_k)
        ]

def _build_session(args) -> Optional[Session]:
    interactive_ok = sys.stdin.isatty()
    attempts = 0
    while True:
        if args.db_type and attempts == 0:
            db_type = args.db_type.lower()
            params = {
                k: v
                for k, v in {
                    "db_path": args.db_path,
                    "host": args.host,
                    "port": args.port,
                    "dbname": args.database,
                    "database": args.database,
                    "user": args.user,
                    "password": args.password,
                }.items()
                if v is not None
            }
        else:
            db_type, params = _choose_db_interactively()
        try:
            extractor = get_schema_extractor(db_type, **params)
            schema_dict = extractor.extract_schema()
            console.print(f"[green]✓ Connected to {db_type} database.[/green]")
            break
        except Exception as e:
            console.print(
                Panel(
                    f"Could not connect: {e}\n[dim]Check your host, port, username, password, or database name.[/dim]",
                    style="red",
                )
            )
            if not interactive_ok:
                return None
            retry = Prompt.ask("Retry connection? (y/N)", default="N")
            if retry.strip().lower() not in {"y", "yes"}:
                return None
            args.db_type = None
            attempts += 1
            continue
    model_id = args.model or "Shaurya-Sethi/transqlate-phi4"
    model_id = model_id.replace("\\", "/")
    if model_id == "Shaurya-Sethi/transqlate-phi4":
        console.print(
            Panel(
                f"[bold]Downloading/loading model from Hugging Face Hub:[/bold]\n[cyan]{model_id}[/cyan]\n"
                "This may take a few minutes the first time.",
                style="cyan",
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    orchestrator = SchemaRAGOrchestrator(tokenizer, schema_dict)
    inference = NL2SQLInference(model_dir=model_id)
    table_embs = build_table_embeddings(schema_dict, orchestrator._embed)
    return Session(
        db_type,
        extractor,
        schema_dict,
        tokenizer,
        orchestrator,
        inference,
        table_embs,
    )

def find_sublist_indices(lst, sublst):
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i:i+len(sublst)] == sublst:
            return i, i+len(sublst)-1
    raise ValueError(f"Sublist {sublst} not found in list.\nList: {lst}")

def extract_schema_token_span(prompt_text, tokenizer):
    schema_start_str = "<SCHEMA>"
    schema_end_str = "</SCHEMA>"
    try:
        sch_start_char = prompt_text.index(schema_start_str)
        sch_end_char = prompt_text.index(schema_end_str) + len(schema_end_str)
    except ValueError:
        raise ValueError("Could not find <SCHEMA> or </SCHEMA> in the prompt text.")
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc['offset_mapping']
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
            f"Could not map <SCHEMA> or </SCHEMA> to token positions. "
            f"sch_start_token_idx: {sch_start_token_idx}, sch_end_token_idx: {sch_end_token_idx}, offsets: {offsets}"
        )
    schema_tokens = enc['input_ids'][sch_start_token_idx : sch_end_token_idx + 1]
    return schema_tokens, sch_start_token_idx, sch_end_token_idx

def _run_model(session: Session, question: str) -> Tuple[str, str]:
    prompt, prompt_ids, _ = session.orchestrator.build_prompt(question)
    try:
        schema_tokens, _, _ = extract_schema_token_span(prompt, session.tokenizer)
    except Exception as e:
        print("[DEBUG] Failed to find <SCHEMA> or </SCHEMA> using offset mapping.")
        print(f"[DEBUG] prompt: {prompt[:500]}...")
        print(f"[DEBUG] prompt_ids: {prompt_ids}")
        raise
    cot_text, sql_text = session.inference.generate(question, schema_tokens)
    return cot_text, sql_text

def _print_result(session: Session, question: str, cot_text: str, sql_text: str, run_sql: bool):
    console.print("\n[dim]Chain of Thought:[/dim]")
    if cot_text:
        cleaned_cot = _fix_cot_dbms(cot_text, session.db_type)
    else:
        cleaned_cot = "[italic dim]Model produced no CoT[/italic dim]"
    console.print(cleaned_cot)
    best_sql = extract_sql(sql_text, cot_text)
    if best_sql:
        console.print("\n[bold cyan]SQL:[/bold cyan]")
        console.print(best_sql, style="bold cyan")
        session.history.append((question, best_sql))
        if run_sql:
            session.execute_sql(best_sql)

def repl(session: Session, run_sql: bool):
    console.print(
        Panel(
            "[bold cyan]Transqlate[/bold cyan] – Natural Language → SQL",
            title="Welcome",
            expand=False,
        )
    )
    console.print("Type your natural language query or :help for commands.\n")
    while True:
        try:
            line = Prompt.ask("[bold green]Transqlate ›[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Exiting…[/red]")
            break
        if not line.strip():
            continue
        if line.startswith(":"):
            cmd, *rest = line[1:].strip().split()
            if cmd in {"exit", "quit", "q"}:
                break
            elif cmd == "help":
                console.print(
                    ":help – show this help\n"
                    ":history – show past queries\n"
                    ":show schema – print formatted schema\n"
                    ":run – re-run last SQL against DB\n"
                    ":examples – sample NL prompts\n"
                    ":clear – clear screen\n"
                    ":change_db – switch to a new database connection\n"
                    ":exit – quit",
                    style="cyan",
                )
            elif cmd == "history":
                for i, (q, s) in enumerate(session.history[-10:], 1):
                    console.print(f"[yellow]{i}.[/yellow] {q} → [cyan]{s}[/cyan]")
            elif cmd == "show" and rest and rest[0] == "schema":
                console.print(format_schema(session.schema_dict))
            elif cmd == "run":
                if not session.history:
                    console.print("[yellow]No previous query to run.[/yellow]")
                else:
                    session.execute_sql(session.history[-1][1])
            elif cmd == "examples":
                console.print(
                    "- Show me total sales by month in 2023\n"
                    "- List top 5 customers by revenue\n"
                    "- Average delivery time per city",
                    style="dim",
                )
            elif cmd == "clear":
                console.clear()
            elif cmd == "change_db":
                console.print(
                    Panel(
                        "[yellow]Changing database connection. Your session history will be cleared.[/yellow]",
                        style="yellow",
                    )
                )
                new_db_type, new_params = _choose_db_interactively()
                try:
                    new_extractor = get_schema_extractor(new_db_type, **new_params)
                    new_schema_dict = new_extractor.extract_schema()
                    new_tokenizer = session.tokenizer
                    new_orchestrator = SchemaRAGOrchestrator(new_tokenizer, new_schema_dict)
                    new_inference = session.inference
                    new_table_embs = build_table_embeddings(new_schema_dict, new_orchestrator._embed)
                    session.db_type = new_db_type
                    session.extractor = new_extractor
                    session.schema_dict = new_schema_dict
                    session.orchestrator = new_orchestrator
                    session.table_embs = new_table_embs
                    session.history = []
                    console.print(f"[green]✓ Switched to new {new_db_type} database.[/green]")
                except Exception as e:
                    _print_exception(e)
                    console.print("[red]Failed to change database. Connection unchanged.[/red]")
            else:
                console.print(f"[red]Unknown command[/red]: {cmd}")
            continue
        try:
            with console.status("[bold cyan]Reasoning...[/bold cyan]", spinner="dots"):
                cot_text, sql_text = _run_model(session, line)
            _print_result(session, line, cot_text, sql_text, run_sql)
        except Exception as e:
            msg = str(e)
            m = re.search(
                r"(?:no such column|Unknown column|column\s+)(?:\s|')(\w+)", msg, re.I
            )
            if m:
                token = m.group(1)
                sugg = session.suggest_schema_terms(token)
                if sugg:
                    console.print(
                        Panel(
                            f"Could not find [bold]{token}[/bold]. Did you mean: "
                            + ", ".join(f"[cyan]{s}[/cyan]" for s in sugg)
                            + "?",
                            style="yellow",
                        )
                    )
                    new_tok = Prompt.ask("Correction", default=sugg[0])
                    if new_tok.strip() and new_tok != token:
                        line = line.replace(token, new_tok)
                        session.history.append((f"(ambig {token}->{new_tok}) {line}", ""))
                        continue
            _print_exception(e)

def oneshot(session: Session, question: str, execute: bool):
    with console.status("[bold cyan]Reasoning...[/bold cyan]", spinner="dots"):
        cot_text, sql_text = _run_model(session, question)
    console.print(Panel(f"[bold green]Query:[/bold green] {question}"))
    _print_result(session, question, cot_text, sql_text, execute)

def main():
    parser = argparse.ArgumentParser("transqlate – Natural Language to SQL CLI")
    parser.add_argument("--interactive", action="store_true", help="Run REPL")
    parser.add_argument("--question", "-q", help="One-shot natural language question")
    parser.add_argument(
        "--execute", action="store_true", help="Execute generated SQL and show results"
    )
    parser.add_argument("--db-type", choices=["sqlite", "postgres", "mysql", "mssql", "oracle"])
    parser.add_argument("--db-path")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--model", help="Path or HF repo for fine-tuned model directory")
    parser.add_argument(
        "--tracebacks",
        action="store_true",
        help="Display Python tracebacks for errors (debugging).",
    )
    args = parser.parse_args()
    global SHOW_TRACEBACKS
    SHOW_TRACEBACKS = args.tracebacks
    if not args.interactive and not args.question:
        parser.error("Provide --interactive or --question/-q.")
    session = _build_session(args)
    if session is None:
        sys.exit(1)
    if args.interactive:
        repl(session, run_sql=args.execute)
    else:
        oneshot(session, question=args.question, execute=args.execute)

if __name__ == "__main__":
    main()
