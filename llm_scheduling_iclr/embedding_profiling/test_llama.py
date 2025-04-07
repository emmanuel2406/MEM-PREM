import json
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Load the model and tokenizer
model = LLM("meta-llama/Meta-Llama-3-8B-Instruct", dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load JSON data from file
with open('./llm_scheduling_iclr/alpaca_data_cleaned.json', 'r') as file:
    data = json.load(file)

print(f"total record~{datetime.now()}: {len(data)}")
i = 0
# Iterate over each prompt in the JSON data
for item in data:
    if i<28650:
        i += 1
        continue
    if i >= 50000:
        break
    try:
        print(f"record~{datetime.now()}: {i}")
        instruction = item['instruction']
        input_text = item['input']

        # Check if input field is empty and format the prompt accordingly
        if input_text.strip():
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Input: {input_text} ### Response:"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Response:"

        messages = [{"role": "user", "content": {prompt}}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Formatted prompt~{datetime.now()}: {formatted_prompt}")
        output = model.generate(formatted_prompt, SamplingParams(max_tokens=512, temperature=0))
        print(f"Output~{datetime.now()}: {output}")
        len(formatted_prompt)
    except Exception as e:
        print(f"Error processing record~{datetime.now()}: {i}, {e}")
    i += 1
print(f"total record processed~{datetime.now()}: {i}")