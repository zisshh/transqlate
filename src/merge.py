from peft import AutoPeftModelForCausalLM

# 1. Path to your checkpoint dir with adapter weights and config
adapter_dir = "model/phi4-transqlate-qlora/checkpoint-8334"

# 2. Load the PEFT adapter (this auto-downloads the base model if missing)
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir,
    torch_dtype="bfloat16",   # Or "float16" if your GPU/CPU does not support bfloat16
    device_map="auto"          # Change to "auto" if you have a big enough GPU
)

# 3. Merge adapter into base model (this "burns in" the LoRA weights)
model = model.merge_and_unload()

# 4. Save to a new standalone directory
model.save_pretrained("phi4-transqlate-qlora-merged")

print("Model merged and saved to 'phi4-transqlate-qlora-merged'")