# src/finetune.py
# Fine-tune Microsoft Phi-4 Mini-Instruct (3.8 B) for NL→SQL with Unsloth QLoRA
# Hardware target: 1× NVIDIA L4 (24 GB VRAM)

"""
This script enables memory-efficient attention using xformers or
flash_attn.

Requires either `xformers` or `flash_attn` to be installed (see requirements.txt).
"""

import os
# ───────────────────────────────────────────────
# Patch for Unsloth + PyTorch 2.5 compatibility:
# Unsloth’s torch.compile integration can crash on PyTorch 2.5 due to
# a Triton flag type change ("triton.multi_kernel" should be int, not bool).
# This disables Unsloth's compilation so it runs in standard PyTorch mode.
# (No significant speed loss, and fully supported by Unsloth devs.)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
os.environ["TMPDIR"] = "/tmp"

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid deadlock warnings
# ───────────────────────────────────────────────

from pathlib import Path

base_dir = Path(__file__).resolve().parent

# -------- prompt formatter MUST be vectorised ----------------------------
def format_prompt(batch):
    """
    Receives a *batch* (dict of lists) and must return a list[str]
    with the SAME length as the batch.
    """
    instructions = batch["instruction"]
    inputs        = batch["input"]
    outputs       = batch["output"]

    # Build one prompt per example
    return [
        f"{inst}\n{inp}\n\n{out}"
        for inst, inp, out in zip(instructions, inputs, outputs)
    ]
# -------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import torch
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from transformers import TrainingArguments
    
    try:
        import xformers
    except ImportError:
        raise ImportError(
            "xformers is required to run this script. Please install it following requirements.txt"
        )
    
    # ----------------------------- #
    # 1  Model & tokenizer loading   #
    # ----------------------------- #
    MAX_SEQ_LEN = 2048
    DTYPE = torch.bfloat16
    BASE_MODEL = "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        dtype=DTYPE,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        device_map="auto",
        attn_implementation="flash_attention_2",  # use PyTorch SDPA if unavailable; xformers acceleration enabled automatically
    )

    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

    # ----------------------------- #
    # 2  Attach LoRA adapters        #
    # ----------------------------- #
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing = "unsloth",
        offload_grads             = False,
    )
    # Print trainable parameters using the PEFT model instance method (Unsloth/PEFT compatible)
    model.print_trainable_parameters()

    # ----------------------------- #
    #  3 Dataset (no pre-tokenisation)  #
    # ----------------------------- #
    data_dir = base_dir.parent / "data"
    raw_ds    = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    # ----------------------------- #
    # 4  Training hyper-parameters  #
    # ----------------------------- #
    BATCH_SIZE = 6
    ACC_STEPS = 8
    LEARNING_RATE = 2e-4
    EPOCHS = 1
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01

    training_args = TrainingArguments(
        output_dir=str(base_dir.parent / "model" / "phi4-transqlate-qlora"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACC_STEPS,
        eval_strategy="steps",
        eval_steps=2500,
        save_strategy="steps",
        save_steps=2500,
        logging_steps=50,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_ds["train"],
        eval_dataset=raw_ds["validation"],
        formatting_func=format_prompt,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        gradient_checkpointing=False,
        args=training_args,
    )

    # ----------------------------- #
    # 5  Train & save               #
    # ----------------------------- #
    trainer.train()
    output_path = base_dir.parent / "model" / "phi4-transqlate-qlora"
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)