# src/inference.py
# ---------------------------------------------------------------------
# Lightweight inference wrapper for a PEFT-QLoRA Phi-4-mini model.
#   • Loads the merged adapter weights saved with trainer.model.save_pretrained().
#   • Builds the exact prompt format used during fine-tuning.
#   • Provides a .generate() helper that returns (cot_text, sql_text).
# ---------------------------------------------------------------------

from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

PROMPT_TEMPLATE = (
    "Translate this question to SQL: {question}\n"
    "Schema:\n{schema}\n\n"
)  # <- identical to fine-tune dataset (PUT_QUESTION_IN_INSTRUCTION=True)

_DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
    do_sample=True,
)

_SQL_SPLIT_RE = re.compile(r"\bSQL\s*:\s*", re.IGNORECASE)


class NL2SQLInference:
    """
    Thin wrapper around a merged-weights Phi-4-mini model.

    Parameters
    ----------
    model_dir : str | Path
        Directory produced by `trainer.model.save_pretrained(...)`.
    device_map : str | dict, optional
        Passed straight to Transformers.  Defaults to "auto".
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "model/phi4-transqlate-qlora",
        device_map: Union[str, dict] = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Ensure forward slashes for HuggingFace repo IDs (remote only)
        def hf_model_id(model_id):
            return str(model_id).replace("\\", "/")
        model_id_str = str(model_dir)
        # Try to resolve as local directory
        if Path(model_id_str).exists():
            model_path = str(Path(model_id_str).resolve())
        else:
            model_path = hf_model_id(model_id_str)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Ensure clean decoding
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_str,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, schema_tokens: List[int]) -> str:
        schema_text = self.tokenizer.decode(
            schema_tokens, skip_special_tokens=False
        )
        return PROMPT_TEMPLATE.format(question=question, schema=schema_text)

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        schema_tokens: List[int],
        **gen_kwargs,
    ) -> Tuple[str, str]:
        """
        Blocking call – returns when generation is finished.

        Returns
        -------
        cot_text : str
            Chain-of-thought produced by the model (may be empty).
        sql_text : str
            Final SQL string (empty if the model didn't emit any).
        """
        prompt = self._build_prompt(question, schema_tokens)
        tokens = self.tokenizer(
            prompt, return_tensors="pt"
        )
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        out_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **{**_DEFAULT_GEN_KWARGS, **gen_kwargs},
        )[0]


        # Strip the prompt part
        gen_ids = out_ids[input_ids.size(1) :]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        # --- PATCHED SECTION BELOW ---
        _SQL_SPLIT_RE = re.compile(r"\bSQL\s*:\s*", re.IGNORECASE)
        _SCHEMA_SPLIT_RE = re.compile(r"(Schema:|<SCHEMA>)", re.IGNORECASE)

        m = _SQL_SPLIT_RE.search(gen_text)
        if m:
            cot_text = gen_text[: m.start()].strip()
            rest = gen_text[m.end():].strip()
            # Remove anything after Schema: or <SCHEMA>
            schema_match = _SCHEMA_SPLIT_RE.search(rest)
            if schema_match:
                sql_text = rest[:schema_match.start()].strip()
            else:
                sql_text = rest.strip()
        else:
            cot_text, sql_text = gen_text.strip(), ""

        return cot_text, sql_text

    # ------------------------------------------------------------------
    def infer_stream(
        self,
        question: str,
        schema_tokens: List[int],
        **gen_kwargs,
    ) -> Iterable[str]:
        """
        Streaming generator – yields decoded substrings as they arrive.
        Retains the same signature the old CLI expected, so you can
        switch back to streaming later if desired.
        """
        prompt = self._build_prompt(question, schema_tokens)
        tokens = self.tokenizer(
            prompt, return_tensors="pt"
        )
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(
                input_ids=input_ids,
                attention_mask=attention_mask,  # <-- Add this!
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{**_DEFAULT_GEN_KWARGS, **gen_kwargs},
            ),
        )
        thread.start()
        for token in streamer:
            yield token
        thread.join()