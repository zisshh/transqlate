# Transqlate Changelog

> **Latest release:** See the [full list of merged PRs](https://github.com/Shaurya-Sethi/transqlate-phi4/pulls?q=is%3Apr+is%3Aclosed+merged%3A%3E%3D2024-06-13&sort=updated&order=desc)

**Period covered:** 36 merged PRs since June 13, 2025

---

## Added

- **Cross-Dialect SQL Post-Processing**
  - Modular post-processing layer now auto-converts model SQL (SQLite/MySQL-style) to the target dialect (MSSQL, PostgreSQL, MySQL, Oracle) before execution.
  - Clear notices are shown for every transformation (e.g., “Adapted from SQLite → T-SQL for SQL Server.”).

- **New CLI Commands and UX**
  - `:edit`: Modify the last generated SQL query before running.
  - `:write`: Compose and run multi-line SQL queries (marked as `(user-written)` in history).
  - `:about`: Placeholder for future documentation.
  - Enhanced `:help` with new commands and detailed descriptions.
  - `:changedb`/`:change_db`: Seamlessly switch database connections mid-session (with full credential recall).
  - `:troubleshoot`: Instant troubleshooting and guidance for connection issues.
  - Password input now uses masked entry (`*` shown if `pwinput` is available).

- **Schema Extraction & Qualification**
  - Schema extractors improved for PostgreSQL, Oracle, and MSSQL:
    - Qualified/escaped table and column identifiers for all supported DBs.
    - Oracle demo/system tables can now be ignored.
    - Safer, user-scoped schema queries for Oracle.

- **Model & Execution Controls**
  - `--max-new-tokens` CLI argument for controlling output length (default 2048).
  - Fine-tuning scripts now support HuggingFace `xformers` for memory-efficient attention.

- **Testing & Reliability**
  - Extensive new and regression tests for SQL extraction, post-processing, schema handling, history, and connection loss scenarios.
  - SQLite fixtures and unit tests for all schema extractors.

---

## Changed

- **SQL Dialect & Post-Processing**
  - MSSQL transformer covers more SQL patterns: `LIMIT`, `TOP`, boolean values, backticks, `TIMESTAMP`, semicolon handling.
  - Avoids duplicate `TOP` clauses.
  - SQL display now always shows both the original and transformed query when post-processing occurs.

- **CLI & User Experience**
  - History entries are now tracked with a status label (e.g., `(edited)`, `(user-written)`).
  - SQL history is better formatted and more informative.
  - `:edit` now finished only with `:finish` (removed `:submit` synonym).
  - Spinner feedback improved for model and database loading.
  - Startup and reconnection spinners improve perceived responsiveness.

- **Error & State Handling**
  - Resilient to DB connection loss, with smarter retry, parameter recall, and error messaging.
  - Error messages are clearer and show tracebacks only if requested via `--tracebacks`.
  - Transactions are rolled back cleanly on errors.
  - CLI disables DDL/DML queries by default for safe execution (DDL/DML require explicit confirmation).

- **Fine-Tuning Workflow**
  - Fine-tuning scripts are now importable (`if __name__ == "__main__"`).
  - Batch size and path logic improved for reliability.
  - Prompt formatting logic split into reusable module.

---

## Fixed

- **SQL Generation & Execution**
  - SQL extraction regex and completeness checks relaxed for better parsing.
  - MSSQL parameter marker bug resolved (`?` for `pyodbc`, `%s` for `pymssql`).
  - Stray semicolons in Oracle queries are now removed before execution to prevent failures.
  - Chain-of-Thought output now correctly reflects the active DB name.

- **Schema Extractors**
  - Oracle extractor now ignores demo/system tables and restricts queries to user-owned objects.
  - Identifier normalization and quoting is robust for all dialects.

- **Model & CLI**
  - SentenceTransformer loading errors are handled gracefully, with user instructions if download fails.
  - Dataset file path issues in training scripts resolved.
  - Duplicate imports and old citation placeholders removed from finetuning scripts.

---

## Removed

- Stray documentation and outdated troubleshooting files.
- Citation placeholders from training scripts.
- Redundant command options (e.g., `:submit` as a synonym for `:finish`).

---

## Highlights

- **Production-Ready, Multi-DB Support:** Transqlate now works seamlessly across SQLite, PostgreSQL, MySQL, MSSQL, and Oracle, with robust schema handling and dialect transformation.
- **CLI-First User Experience:** New commands, interactive history, and resilient reconnection make for a polished CLI suitable for both technical and non-technical users.
- **Fine-Tuned Model Integration:** All pipelines and workflows now align with the latest fine-tuned Phi-4 Mini model.

---

**Full details and previous releases:**  
[Transqlate on GitHub](https://github.com/Shaurya-Sethi/transqlate-phi4)