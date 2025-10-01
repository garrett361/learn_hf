from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Sample SFT data in conversational format
sft_data = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

formatted_text = tok.apply_chat_template(
    sft_data,
    tokenize=False,  # Get string first
    add_generation_prompt=False,  # Don't add prompt for training
)

tokenized = tok(
    formatted_text,
    truncation=False,
    max_length=None,
    padding=False,
    return_tensors="pt",
)

print(f"{formatted_text=}")
print(f"{tokenized=}")
sft_data = [
    {"role": "user", "content": "What is the capital of France?"},
]
