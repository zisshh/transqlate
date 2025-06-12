import sys
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from finetune import format_prompt

def test_format_prompt():
    batch = {
        "instruction": ["Generate SQL"],
        "input": ["List all users"],
        "output": ["SELECT * FROM users;"]
    }
    expected = ["Generate SQL\nList all users\n\nSELECT * FROM users;"]
    assert format_prompt(batch) == expected
