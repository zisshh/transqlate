"""Simpler wrapper used by unit tests for prompt formatting."""

def format_prompt(example: dict) -> dict:
    return {"text": f"{example['instruction']}\n{example['input']}\n\n{example['output']}"}
