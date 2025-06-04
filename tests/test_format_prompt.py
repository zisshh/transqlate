import sys
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from finetune_phi4_nl2sql import format_prompt

def test_format_prompt():
    example = {
        "instruction": "Generate SQL",
        "input": "List all users",
        "output": "SELECT * FROM users;",
    }
    expected = {
        "text": "Generate SQL\nList all users\n\nSELECT * FROM users;"
    }
    assert format_prompt(example) == expected
