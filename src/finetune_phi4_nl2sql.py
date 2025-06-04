# Fine-tune Microsoft Phi-4 Mini-Instruct (3.8 B) for NL→SQL with Unsloth QLoRA
# Hardware target: 1× NVIDIA L4 (24 GB VRAM)




def format_prompt(example):
    return {
        "text": f"{example['instruction']}\n{example['input']}\n\n{example['output']}"
    }


if __name__ == "__main__":
    import torch
    from unsloth import FastLanguageModel, SFTTrainer
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from transformers import TrainingArguments

    # ----------------------------- #
    # 1  Model & tokenizer loading   #
    # ----------------------------- #
    MAX_SEQ_LEN        = 2048               # fits easily in 24 GB with QLoRA
    DTYPE              = torch.bfloat16     # L4 supports bf16 natively
    BASE_MODEL         = "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit"   # 3.8 B

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name           = BASE_MODEL,
        dtype                = DTYPE,
        max_seq_length       = MAX_SEQ_LEN,
        load_in_4bit         = True,        # QLoRA
        use_flash_attention_2= True,        # saves ⇢ 15-20 % VRAM
        device_map           = "auto",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

    # ----------------------------- #
    # 2  Attach LoRA adapters        #
    # ----------------------------- #
    model = FastLanguageModel.get_peft_model(
        model,
        r               = 64,               # rank
        lora_alpha      = 128,              # 2× r
        lora_dropout    = 0.05,
        target_modules  = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        bias            = "none",
    )
    FastLanguageModel.print_trainable_parameters(model)  # sanity-check

    # ----------------------------- #
    # 3  Dataset loading & mapping   #
    # ----------------------------- #
    data_files = {"train": "train.jsonl", "validation": "val.jsonl"}
    raw_ds     = load_dataset("json", data_files=data_files)

    ds = raw_ds.map(format_prompt, remove_columns=raw_ds["train"].column_names)

    # ----------------------------- #
    # 4  Training hyper-parameters   #
    # ----------------------------- #
    #   GPU    : L4 24 GB (bf16 + 4-bit)
    #   Batch  : 6 × 2 K tokens in VRAM  → ≈9 K tokens
    #   Acc.Steps : 8 → Eff. batch ≈48 K tokens / step
    BATCH_SIZE    = 6
    ACC_STEPS     = 8
    LEARNING_RATE = 2e-4
    EPOCHS        = 1
    WARMUP_RATIO  = 0.03
    WEIGHT_DECAY  = 0.01

    training_args = TrainingArguments(
        output_dir                 = "../model/phi4-transqlate-qlora",
        per_device_train_batch_size= BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps= ACC_STEPS,
        evaluation_strategy        = "steps",
        eval_steps                 = 2500,
        save_strategy              = "steps",
        save_steps                 = 2500,
        logging_steps              = 50,
        num_train_epochs           = EPOCHS,
        learning_rate              = LEARNING_RATE,
        warmup_ratio               = WARMUP_RATIO,
        lr_scheduler_type          = "cosine",
        weight_decay               = WEIGHT_DECAY,
        bf16                       = True,
        gradient_checkpointing     = True,
        max_grad_norm              = 1.0,
        optim                      = "adamw_8bit",  # memory-efficient
        report_to                  = "tensorboard",        # set to "wandb" if you use it
    )

    trainer = SFTTrainer(
        model                   = model,
        tokenizer               = tokenizer,
        train_dataset           = ds["train"],
        eval_dataset            = ds["validation"],
        dataset_text_field      = "text",
        max_seq_length          = MAX_SEQ_LEN,
        args                    = training_args,
    )

    # ----------------------------- #
    # 5  Train & save               #
    # ----------------------------- #
    trainer.train()
    trainer.model.save_pretrained("../model/phi4-transqlate-qlora")
    tokenizer.save_pretrained("../model/phi4-transqlate-qlora")
