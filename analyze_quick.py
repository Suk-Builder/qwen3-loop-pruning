import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle

MODEL_PATH = "dynamic_infinite_train_v2/final"
OUTPUT_DIR = "./pruning_analysis"
NUM_QUESTIONS = 20
MAX_NEW_TOKENS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

TEST_QUESTIONS = [
    "什么是量子力学？", "Explain relativity.", "如何学习编程？",
    "What is AI?", "请解释区块链。", "How does photosynthesis work?",
    "什么是民主？", "Explain neural networks.", "如何保持健康？",
    "What is gravity?", "什么是通货膨胀？", "How do vaccines work?",
    "请评价莎士比亚。", "What is dark matter?", "如何管理时间？",
    "Explain evolution.", "什么是社会主义？", "How do computers work?",
    "请分析经济形势。", "What is entropy?",
]

num_layers = len(model.model.layers)
intermediate_size = model.config.intermediate_size
activation_counts = [np.zeros(intermediate_size, dtype=np.int64) for _ in range(num_layers)]
total_tokens = 0

hooks = []

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        global total_tokens
        activated = (output > 0).sum(dim=(0, 1))
        activation_counts[layer_idx] += activated.cpu().numpy()
        total_tokens += output.shape[0] * output.shape[1]
    return hook_fn

for i, layer in enumerate(model.model.layers):
    handle = layer.mlp.act_fn.register_forward_hook(make_hook(i))
    hooks.append(handle)

print(f"Running {NUM_QUESTIONS} questions...")

for q_idx, question in enumerate(TEST_QUESTIONS[:NUM_QUESTIONS]):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"  Q{q_idx+1}/{NUM_QUESTIONS} done, tokens={total_tokens}")

for h in hooks:
    h.remove()

print(f"\nTotal tokens: {total_tokens}")

frequencies = []
for layer_idx in range(num_layers):
    freq = activation_counts[layer_idx] / max(total_tokens, 1)
    frequencies.append(freq)
    alive = (freq > 0.01).sum()
    print(f"Layer {layer_idx:2d}: alive(>1%)={alive}/{intermediate_size} ({100*alive/intermediate_size:.1f}%)")

results = {
    "activation_counts": activation_counts,
    "frequencies": frequencies,
    "total_tokens": total_tokens,
    "num_layers": num_layers,
    "intermediate_size": intermediate_size,
}

with open(os.path.join(OUTPUT_DIR, "activation_analysis.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved to {OUTPUT_DIR}/activation_analysis.pkl")

all_freqs = np.concatenate(frequencies)
print(f"Neurons with freq<0.01: {(all_freqs < 0.01).sum()} / {len(all_freqs)}")
print(f"Neurons with freq<0.05: {(all_freqs < 0.05).sum()} / {len(all_freqs)}")
print(f"Neurons with freq<0.1: {(all_freqs < 0.1).sum()} / {len(all_freqs)}")
