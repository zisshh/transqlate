# AGENTS.md Guide for transqlate

This file provides guidance for all AI coding assistants and agents working in this repository.
Follow the instructions below for code style, structure, and project workflow.

## 1. Project Structure & Navigation
- **`/src`**: Main source code for the `transqlate` package.
  - **`/transqlate/cli`**: Handles the Command Line Interface (CLI) entry point and all user interaction logic.
  - **`/transqlate/schema_pipeline`**: Contains modules for schema extraction, formatting, and Retrieval-Augmented Generation (RAG). This is the core of the SQL translation logic.
  - **`/transqlate/inference.py`**: Contains the model inference logic.
- **`/tests`**: All unit and integration tests. New tests must be added here for any new features or bug fixes.
- **`/assets`**: Project assets like images and GIFs. Do not modify unless specified.
- **`/model`**: Contains fine-tuned model checkpoints. Do not modify this directory.
- **`/benchmark`**: Scripts, data, and results for benchmarking model performance.
- **`/dataset`**: Contains training and validation data, typically in JSONL format.

## 2. Key Patterns & Coding Conventions
- **Language & Style**: All new code must be Python 3.8+ and strictly follow PEP 8 standards. Use `snake_case` for functions and variables, and `PascalCase` for classes.
- **Type Hinting**: Add type hints to all function and method signatures for clarity and static analysis.
- **Docstrings**: Write clear and comprehensive docstrings for all public modules, classes, and functions.
- **Immutability**: Avoid global variables. Pass state explicitly through function arguments or class attributes.
- **API Usage**: Model inference must use the Hugging Face Transformers API, as demonstrated in `src/transqlate/inference.py`.
- **Guardrails**: All SQL execution that alters data must first confirm user intent. See existing CLI logic for implementation examples.
- **Schema Qualification**: When generating SQL for Postgres, SQL Server, or Oracle, always qualify table names with their schema if it is not the default schema.

## 3. Developer Workflows
- **Installation**: Install all required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- **Running the CLI**: After installation, the main tool can be run interactively with:
  ```bash
  transqlate --interactive
  ```
- **Testing**: Run the entire test suite using `pytest` from the root directory. All changes must pass existing tests, and new functionality requires new tests.
  ```bash
  pytest
  ```
- **Model Fine-tuning**: Fine-tuning requires a GCP environment with a GPU. Run the script via:
  ```bash
  python src/finetune.py
  ```
- **Benchmarking**: To evaluate model performance, use the scripts located in the `/benchmark/scripts/` directory. Refer to `benchmark/README.md` for detailed commands.

## 4. Integration & External Dependencies
- **Model Loading**: Models are loaded from the Hugging Face Hub.
- **Embeddings**: The `sentence-transformers` library is used for generating sentence embeddings and is downloaded on its first run.
- **Quantization**: Loading 4-bit quantized models requires a CUDA-enabled GPU and the `bitsandbytes` library.

## 5. Pull Requests & Commits
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for clear and standardized commit messages (e.g., `feat:`, `fix:`, `docs:`, `test:`).
- **Pull Requests**: PR messages must include a summary of changes, a reference to any related issues, and confirmation that all tests pass.

## 6. Scope and Precedence
- This file applies to all subfolders unless a more-specific AGENTS.md file is present.
- User/system prompt instructions override anything in AGENTS.md.
