import os
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

MODEL_PATH = "dynamic_infinite_train_v2/final"
ANALYSIS_PATH = "./pruning_analysis/activation_analysis.pkl"
OUTPUT_DIR = "./pruned_model_50pct"
PRUNE_RATIO = 0.50  # 剪掉50%

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading analysis results...")
with open(ANALYSIS_PATH, "rb") as f:
    analysis = pickle.load(f)

frequencies = analysis["frequencies"]
num_layers = analysis["num_layers"]
intermediate_size = analysis["intermediate_size"]

print(f"Loading original model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 先在CPU上操作，避免GPU内存不够
    attn_implementation="sdpa",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Original model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# 对每层决定保留哪些神经元
keep_indices_per_layer = []
new_intermediate_size = None

for layer_idx in range(num_layers):
    freq = frequencies[layer_idx]
    keep_count = int(intermediate_size * (1 - PRUNE_RATIO))
    # 保留激活频率最高的 keep_count 个神经元
    keep_indices = np.argsort(freq)[-keep_count:]
    keep_indices = np.sort(keep_indices)  # 保持顺序，避免权重重排
    keep_indices_per_layer.append(keep_indices)

    alive_ratio = (freq > 0).sum() / intermediate_size
    print(f"Layer {layer_idx:2d}: keep {keep_count}/{intermediate_size} neurons, "
          f"alive_ratio={alive_ratio:.2%}")

new_intermediate_size = keep_count
print(f"\nNew intermediate_size: {new_intermediate_size} (was {intermediate_size})")

# 修改config
model.config.intermediate_size = new_intermediate_size

# 重建每层的MLP
for layer_idx in range(num_layers):
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    keep = keep_indices_per_layer[layer_idx]

    old_gate = mlp.gate_proj.weight.data  # [intermediate_size, hidden_size]
    old_up = mlp.up_proj.weight.data      # [intermediate_size, hidden_size]
    old_down = mlp.down_proj.weight.data  # [hidden_size, intermediate_size]

    # 创建新的线性层
    new_gate = nn.Linear(model.config.hidden_size, new_intermediate_size, bias=False)
    new_up = nn.Linear(model.config.hidden_size, new_intermediate_size, bias=False)
    new_down = nn.Linear(new_intermediate_size, model.config.hidden_size, bias=False)

    # 复制权重
    new_gate.weight.data = old_gate[keep, :].clone()
    new_up.weight.data = old_up[keep, :].clone()
    new_down.weight.data = old_down[:, keep].clone()

    # 替换
    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down

    print(f"  Layer {layer_idx}: gate {old_gate.shape} -> {new_gate.weight.shape}, "
          f"down {old_down.shape} -> {new_down.weight.shape}")

# 保存
print(f"\nSaving pruned model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

new_params = sum(p.numel() for p in model.parameters())
old_params = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
    attn_implementation="sdpa", trust_remote_code=True
).parameters())

print(f"\nOriginal params: {old_params/1e6:.1f}M")
print(f"Pruned params: {new_params/1e6:.1f}M")
print(f"Reduction: {(1 - new_params/old_params)*100:.1f}%")
print(f"Saved to {OUTPUT_DIR}")
