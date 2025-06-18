# src/inference.py
# ---------------------------------------------------------------------
# Lightweight inference wrapper for a PEFT-QLoRA Phi-4-mini model.
#   • Loads the merged adapter weights saved with trainer.model.save_pretrained().
#   • Builds the exact prompt format used during fine-tuning.
#   • Provides a .generate() helper that returns (cot_text, sql_text).
# ---------------------------------------------------------------------

from __future__ import annotations
import os
import re
import threading
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

from transqlate.utils.hardware import detect_device_and_quant

PROMPT_TEMPLATE = (
    "Translate this question to SQL: {question}\n"
    "Schema:\n{schema}\n\n"
)  # <- identical to fine-tune dataset (PUT_QUESTION_IN_INSTRUCTION=True)

_DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=2048,
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
        **kwargs,
    ):
        no_quant_flag = kwargs.get("quantization") is False
        env_opt_out = os.getenv("TRANSQLATE_NO_QUANT")

        device_map, dtype, use_4bit = detect_device_and_quant(
            no_quant_flag or bool(env_opt_out)
        )

        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.use_4bit = use_4bit
        self.device_map = device_map
        self.dtype = dtype

        # Ensure forward slashes for HuggingFace repo IDs (remote only)
        def hf_model_id(model_id):
            return str(model_id).replace("\\", "/")
        model_id_str = str(model_dir)
        # Try to resolve as local directory
        if Path(model_id_str).exists():
            model_path = str(Path(model_id_str).resolve())
        else:
            model_path = hf_model_id(model_id_str)

        try:
            config = AutoConfig.from_pretrained(model_path)
        except Exception:
            config = PretrainedConfig()
        if hasattr(config, "quantization_config") and config.quantization_config is None:
            delattr(config, "quantization_config")
            config = config.__class__.from_dict(config.to_dict())

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Ensure clean decoding
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "config": config,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "You passed `quantization_config` or equivalent parameters to "
                    "`from_pretrained` but the model you're loading already has a "
                    "`quantization_config` attribute. The `quantization_config` "
                    "from the model will be used."
                ),
                category=UserWarning,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_str,
                **model_kwargs,
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