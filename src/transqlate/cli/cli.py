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

try:
    from pwinput import pwinput  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    pwinput = None

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
from transqlate.embedding_utils import (
    EmbeddingDownloadError,
    load_sentence_embedder,
)

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

# Database-specific troubleshooting instructions
_DB_TROUBLESHOOT = {
    "mssql": "\n".join(
        [
            "1. Enable SQL Server Authentication in SSMS (Server > Properties > Security).",
            "2. Set a password for the `sa` account and ensure it is enabled (Security > Logins > sa).",
            "3. Enable TCP/IP via SQL Server Configuration Manager and set port 1433 (Protocols for SQLEXPRESS).",
            "4. Restart the SQL Server service after making changes.",
            "5. Allow TCP port 1433 through the firewall if connecting remotely.",
        ]
    )
}

# Toggle for showing Python tracebacks with errors. Can be enabled via
# the --tracebacks CLI flag during debugging.
SHOW_TRACEBACKS = False


def _positive_int(value: str) -> int:
    """Argparse helper to ensure a positive integer."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"max_new_tokens must be a positive integer, got '{value}'"
        )

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


_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def _post_process_mssql_sql(sql: str) -> str:
    """Transform common SQLite/MySQL syntax to T–SQL.

    Parameters
    ----------
    sql : str
        Input SQL string potentially using other dialect syntax.

    Returns
    -------
    str
        SQL string adjusted for MSSQL.

    Examples
    --------
    >>> _post_process_mssql_sql('SELECT "name" FROM users LIMIT 5;')
    'SELECT TOP 5 [name] FROM users;'
    >>> _post_process_mssql_sql('SELECT * FROM t WHERE done=TRUE;')
    'SELECT * FROM t WHERE done=1;'
    """

    out = sql

    # IDENTIFIERS
    out = re.sub(r'"([A-Za-z0-9_]+)"', r'[\1]', out)

    # BOOLEAN LITERALS
    out = re.sub(r"\bTRUE\b", "1", out, flags=re.IGNORECASE)
    out = re.sub(r"\bFALSE\b", "0", out, flags=re.IGNORECASE)

    # DATA TYPES
    out = re.sub(r"\bDATETIME\b", "DATETIME2", out, flags=re.IGNORECASE)
    out = re.sub(r"\bINTEGER\s+PRIMARY\s+KEY\b", "INT PRIMARY KEY", out, flags=re.IGNORECASE)
    out = re.sub(r"\bAUTOINCREMENT\b", "IDENTITY(1,1)", out, flags=re.IGNORECASE)

    # LIMIT -> TOP
    m = _LIMIT_RE.search(out)
    if m and "TOP" not in out.upper():
        n = m.group(1)
        out = _LIMIT_RE.sub("", out)
        if re.search(r"(?i)SELECT\s+DISTINCT", out):
            out = re.sub(r"(?i)SELECT\s+DISTINCT", f"SELECT DISTINCT TOP {n}", out, count=1)
        else:
            out = re.sub(r"(?i)SELECT", f"SELECT TOP {n}", out, count=1)

    # Cleanup stray spaces before semicolons
    out = re.sub(r"\s+;", ";", out)

    return out.strip()


def _print_troubleshooting(db_type: str) -> None:
    """Display troubleshooting steps for the given DB type."""
    text = _DB_TROUBLESHOOT.get(db_type.lower())
    if not text:
        text = (
            "Check your host, port, username, password and database name. "
            "Ensure the server is reachable."
        )
    console.print(Panel(text, title="Troubleshooting", style="cyan"))

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
            if db_path.strip().lower().lstrip(":") == "troubleshoot":
                _print_troubleshooting(db_type)
                continue
            if (db_path.startswith('"') and db_path.endswith('"')) or (db_path.startswith("'") and db_path.endswith("'")):
                console.print(
                    "[red]Please remove the wrapping quotes from your path and try again.[/red]"
                )
                continue
            params["db_path"] = db_path
            break
    else:
        while True:
            host = Prompt.ask("Host", default="localhost")
            if host.strip().lower().lstrip(":") == "troubleshoot":
                _print_troubleshooting(db_type)
                continue
            params["host"] = host
            break
        default_port = {
            "postgres": "5432",
            "postgresql": "5432",
            "mysql": "3306",
            "mssql": "1433",
            "oracle": "1521",
        }.get(db_type, "5432")
        while True:
            port_val = Prompt.ask("Port", default=default_port)
            if port_val.strip().lower().lstrip(":") == "troubleshoot":
                _print_troubleshooting(db_type)
                continue
            try:
                params["port"] = int(port_val)
                break
            except ValueError:
                console.print("[red]Port must be a number.[/red]")
        if db_type in {"postgres", "postgresql"}:
            while True:
                dbn = Prompt.ask("Database name")
                if dbn.strip().lower().lstrip(":") == "troubleshoot":
                    _print_troubleshooting(db_type)
                    continue
                params["dbname"] = dbn
                break
        else:
            while True:
                dbn = Prompt.ask("Database name")
                if dbn.strip().lower().lstrip(":") == "troubleshoot":
                    _print_troubleshooting(db_type)
                    continue
                params["database"] = dbn
                break
        while True:
            user = Prompt.ask("Username")
            if user.strip().lower().lstrip(":") == "troubleshoot":
                _print_troubleshooting(db_type)
                continue
            params["user"] = user
            break
        console.print("[dim](Your password will not be shown as you type.)[/dim]")
        if pwinput:
            while True:
                pw = pwinput(prompt="Password: ", mask="*")
                if pw.strip().lower().lstrip(":") == "troubleshoot":
                    _print_troubleshooting(db_type)
                    continue
                params["password"] = pw
                break
        else:
            while True:
                pw = getpass("Password: ")
                if pw.strip().lower().lstrip(":") == "troubleshoot":
                    _print_troubleshooting(db_type)
                    continue
                params["password"] = pw
                break
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

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _looks_incomplete(sql: str) -> bool:
    """Heuristically determine if an SQL statement appears incomplete."""
    if not sql or not sql.strip():
        return True
    sql = sql.strip()
    if sql.count("(") > sql.count(")"):
        return True
    if sql.count("'") % 2 == 1 or sql.count('"') % 2 == 1:
        return True
    tail = re.sub(r";\s*$", "", sql).rstrip().upper()
    incomplete_tokens = {
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "ON",
        "AND",
        "OR",
        "GROUP",
        "ORDER",
        "VALUES",
        "SET",
        "INSERT",
        "UPDATE",
        "DELETE",
    }
    if tail.endswith(tuple(incomplete_tokens)):
        return True
    if tail.endswith(','):
        return True
    return False

def _connection_lost(exc: Exception) -> bool:
    """Return True if `exc` appears related to a lost or closed connection."""
    msg = str(exc).lower()
    patterns = [
        "connection already closed",
        "connection is closed",
        "closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection not open",
        "server closed the connection",
        "ssl connection has been closed",
        "ssl syscall error: eof detected",
        "server has gone away",
        "lost connection",
        "not connected",
        "broken pipe",
        "pipe closed",
        "cannot operate on a closed database",
    ]
    return any(p in msg for p in patterns)

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
        connection_params: Optional[dict] = None,
    ):
        self.db_type = db_type
        self.extractor = extractor
        self.schema_dict = schema_dict
        self.tokenizer = tokenizer
        self.orchestrator = orchestrator
        self.inference = inference
        self.history: List[Tuple[str, str]] = []
        self.table_embs = table_embs
        self.connection_params = connection_params or {}

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def reconnect(self) -> bool:
        """Attempt to re-establish the DB connection using stored params."""
        if not self.connection_params:
            return False
        try:
            new_extractor = get_schema_extractor(self.db_type, **self.connection_params)
            new_schema_dict = new_extractor.extract_schema()
            new_orch = SchemaRAGOrchestrator(self.tokenizer, new_schema_dict)
            new_table_embs = build_table_embeddings(new_schema_dict, new_orch._embed)
            # Close old extractor if possible
            try:
                self.extractor.close()
            except Exception:
                pass
            self.extractor = new_extractor
            self.schema_dict = new_schema_dict
            self.orchestrator = new_orch
            self.table_embs = new_table_embs
            console.print("[green]✓ Reconnected to database.[/green]")
            return True
        except Exception as exc:
            _print_exception(exc)
            return False

    def execute_sql(self, sql: str):
        if _looks_incomplete(sql):
            console.print(
                Panel(
                    f"[red]Query appears incomplete and was not executed.[/red]\n[bold yellow]{sql}[/bold yellow]\n[dim]Edit your question or fix the SQL and try again.[/dim]",
                    style="red",
                )
            )
            return
        if self.db_type.lower() == "mssql":
            sql = _post_process_mssql_sql(sql)
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
            if _connection_lost(e):
                console.print(
                    Panel(
                        f"Connection lost while executing query (likely due to idle timeout):\n{msg}",
                        style="red",
                    )
                )
                interactive = sys.stdin.isatty()
                retry_sql = False
                reconnect_ok = False
                if interactive:
                    resp = Prompt.ask(
                        "Reconnect using previous credentials? (Y/n/change)",
                        default="Y",
                    )
                else:
                    resp = "Y"
                if resp.lower().startswith("y"):
                    reconnect_ok = self.reconnect()
                elif resp.lower().startswith("change"):
                    new_db_type, new_params = _choose_db_interactively()
                    self.db_type = new_db_type
                    self.connection_params = new_params
                    reconnect_ok = self.reconnect()
                if reconnect_ok:
                    if interactive:
                        retry = Prompt.ask("Retry last query? (Y/n)", default="Y")
                        retry_sql = retry.lower().startswith("y")
                    else:
                        retry_sql = True
                else:
                    console.print(
                        "[red]Reconnection failed. Use :change_db to configure a new connection.[/red]"
                    )
                if retry_sql and reconnect_ok:
                    try:
                        self.execute_sql(sql)
                    except Exception as e2:
                        _print_exception(e2)
                return
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
            with console.status("[bold cyan]Connecting to database...[/bold cyan]", spinner="dots"):
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
            retry = Prompt.ask(
                "Retry connection? (y/N/troubleshoot)", default="N"
            )
            resp = retry.strip().lower().lstrip(":")
            if resp == "troubleshoot":
                _print_troubleshooting(db_type)
                args.db_type = None
                attempts += 1
                continue
            if resp not in {"y", "yes"}:
                return None
            args.db_type = None
            attempts += 1
            continue
    model_id = args.model or "Shaurya-Sethi/transqlate-phi4"
    model_id = model_id.replace("\\", "/")

    msg = "Loading model from Hugging Face..."
    if not Path(model_id).exists():
        msg = "Downloading model from Hugging Face Hub..."
    with console.status(f"[bold cyan]{msg}[/bold cyan]", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        inference = NL2SQLInference(model_dir=model_id)

    try:
        embed_model = load_sentence_embedder("all-MiniLM-L6-v2")
    except EmbeddingDownloadError as exc:
        console.print(
            Panel(
                "Failed to download embedding model from Hugging Face. "
                "Please check your internet connection, or pre-download the model using:\n"
                "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"\n"
                "The CLI cannot continue without this model.",
                style="red",
            )
        )
        return None

    orchestrator = SchemaRAGOrchestrator(
        tokenizer,
        schema_dict,
        embed_model=embed_model,
    )
    table_embs = build_table_embeddings(schema_dict, orchestrator._embed)
    return Session(
        db_type,
        extractor,
        schema_dict,
        tokenizer,
        orchestrator,
        inference,
        table_embs,
        connection_params=params,
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

def _run_model(session: Session, question: str, max_new_tokens: int) -> Tuple[str, str]:
    prompt, prompt_ids, _ = session.orchestrator.build_prompt(question)
    try:
        schema_tokens, _, _ = extract_schema_token_span(prompt, session.tokenizer)
    except Exception as e:
        print("[DEBUG] Failed to find <SCHEMA> or </SCHEMA> using offset mapping.")
        print(f"[DEBUG] prompt: {prompt[:500]}...")
        print(f"[DEBUG] prompt_ids: {prompt_ids}")
        raise
    cot_text, sql_text = session.inference.generate(
        question, schema_tokens, max_new_tokens=max_new_tokens
    )
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


def _prompt_edit_sql(original_sql: str) -> Optional[str]:
    """Prompt user to edit or replace the provided SQL string.

    Returns the edited SQL, or ``None`` if editing was cancelled.
    """
    console.print(Panel(original_sql, title="Current SQL", style="cyan"))
    console.print(
        "[dim]Enter new SQL. Submit empty line to cancel or type 'CANCEL' on a new line.[/dim]"
    )
    lines: List[str] = []
    while True:
        try:
            new_line = input()
        except EOFError:
            break
        if not lines and new_line.strip() == "":
            return None
        if new_line.strip().upper() == "CANCEL":
            return None
        if new_line == "":
            break
        lines.append(new_line)
    edited = "\n".join(lines).strip()
    if not edited:
        return None
    return edited

def repl(session: Session, run_sql: bool, max_new_tokens: int):
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
                    ":edit – edit last SQL before running\n"
                    ":examples – sample NL prompts\n"
                    ":clear – clear screen\n"
                    ":change_db – switch to a new database connection\n"
                    ":about – about this tool\n"
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
            elif cmd == "edit":
                if not session.history:
                    console.print("[yellow]No SQL query to edit.[/yellow]")
                else:
                    edited = _prompt_edit_sql(session.history[-1][1])
                    if edited is None:
                        console.print("[yellow]Edit cancelled. SQL unchanged.[/yellow]")
                    else:
                        question = session.history[-1][0]
                        session.history.append((question + " [edited]", edited))
                        console.print("[green]SQL updated. Use :run to execute.[/green]")
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
                    session.connection_params = new_params
                    session.history = []
                    console.print(f"[green]✓ Switched to new {new_db_type} database.[/green]")
                except Exception as e:
                    _print_exception(e)
                    console.print("[red]Failed to change database. Connection unchanged.[/red]")
            elif cmd == "about":
                console.print(
                    Panel(
                        "Transqlate CLI allows natural language to SQL translation."\
                        "\nDetailed documentation coming soon.",
                        title="About",
                        style="cyan",
                    )
                )
            else:
                console.print(f"[red]Unknown command[/red]: {cmd}")
            continue
        try:
            with console.status("[bold cyan]Reasoning...[/bold cyan]", spinner="dots"):
                cot_text, sql_text = _run_model(session, line, max_new_tokens)
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

def oneshot(session: Session, question: str, execute: bool, max_new_tokens: int):
    with console.status("[bold cyan]Reasoning...[/bold cyan]", spinner="dots"):
        cot_text, sql_text = _run_model(session, question, max_new_tokens)
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
        "--max-new-tokens",
        type=_positive_int,
        default=2048,
        help=(
            "Maximum tokens to generate per response (default: 2048). "
            "For very complex or verbose queries, increase this limit."
        ),
    )
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
        repl(session, run_sql=args.execute, max_new_tokens=args.max_new_tokens)
    else:
        oneshot(
            session,
            question=args.question,
            execute=args.execute,
            max_new_tokens=args.max_new_tokens,
        )

if __name__ == "__main__":
    main()
