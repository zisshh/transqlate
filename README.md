# Phi-4 QLoRA

This project fine-tunes Microsoft Phi-4 Mini-Instruct with Unsloth's QLoRA.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Place `train_split.jsonl` and `val_split.jsonl` next to `src/finetune_phi4_nl2sql.py` and run:

```bash
python src/finetune_phi4_nl2sql.py
```

The fine-tuned model will be written to `model/phi4-transqlate-qlora`.
