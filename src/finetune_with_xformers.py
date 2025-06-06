# src/finetune_with_xformers.py
"""
This script enables memory-efficient attention using xformers instead of
flash_attn for environments where flash attention is not supported or fails
to install.

Requires `xformers` to be installed (see requirements.txt).
"""

from pathlib import Path

base_dir = Path(__file__).resolve().parent


def format_prompt(example):
    return {
        "text": f"{example['instruction']}\n{example['input']}\n\n{example['output']}"
    }


if __name__ == "__main__":
    from pathlib import Path
    import torch
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from transformers import TrainingArguments
    from transformers.utils import is_xformers_available

    if not is_xformers_available():
        raise ImportError(
            "xformers is required to run this script. Please install it following requirements.txt"
        )
    from transformers import enable_xformers_memory_efficient_attention

    # enable xformers based attention in Transformers
    enable_xformers_memory_efficient_attention()

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
        attn_implementation="sdpa",  # use PyTorch SDPA; xformers acceleration enabled above
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
    )
    FastLanguageModel.print_trainable_parameters(model)

    # ----------------------------- #
    # 3  Dataset loading & mapping   #
    # ----------------------------- #
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"
    data_files = {
        "train": str(data_dir / "train_split.jsonl"),
        "validation": str(data_dir / "val_split.jsonl"),
    }
    raw_ds = load_dataset("json", data_files=data_files)

    ds = raw_ds.map(format_prompt, remove_columns=raw_ds["train"].column_names)

    # ----------------------------- #
    # 4  Training hyper-parameters   #
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
        evaluation_strategy="steps",
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
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    # ----------------------------- #
    # 5  Train & save               #
    # ----------------------------- #
    trainer.train()
    output_path = base_dir.parent / "model" / "phi4-transqlate-qlora"
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
