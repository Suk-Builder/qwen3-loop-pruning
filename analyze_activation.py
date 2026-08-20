import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle

MODEL_PATH = "dynamic_infinite_train_v2/final"
OUTPUT_DIR = "./pruning_analysis"
NUM_QUESTIONS = 100
MAX_NEW_TOKENS = 256

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

# 100个测试问题（混合中英文，覆盖不同领域）
TEST_QUESTIONS = [
    "什么是量子力学？",
    "Explain the theory of relativity.",
    "如何学习编程？",
    "What is the capital of France?",
    "人工智能会取代人类吗？",
    "How does photosynthesis work?",
    "请解释区块链技术。",
    "What causes climate change?",
    "如何做出好吃的红烧肉？",
    "Explain DNA replication.",
    "什么是民主制度？",
    "How do airplanes fly?",
    "请分析当前经济形势。",
    "What is machine learning?",
    "如何保持健康的生活方式？",
    "Explain the Big Bang theory.",
    "什么是马克思主义？",
    "How does the internet work?",
    "请描述一下你的训练过程。",
    "What is consciousness?",
    "如何写好一篇论文？",
    "Explain neural networks.",
    "什么是通货膨胀？",
    "How do vaccines work?",
    "请评价一下莎士比亚。",
    "What is dark matter?",
    "如何管理时间？",
    "Explain evolution by natural selection.",
    "什么是社会主义？",
    "How do computers process information?",
    "请介绍一下中国传统文化。",
    "What is entropy?",
    "如何培养创造力？",
    "Explain the water cycle.",
    "什么是资本主义？",
    "How do earthquakes happen?",
    "请分析一下中美关系的未来。",
    "What is a black hole?",
    "如何提高记忆力？",
    "Explain plate tectonics.",
    "什么是哲学？",
    "How do batteries work?",
    "请描述一下理想的社会。",
    "What is the speed of light?",
    "如何克服拖延症？",
    "Explain the immune system.",
    "什么是艺术？",
    "How do cameras work?",
    "请评价一下现代教育体系。",
    "What is gravity?",
    "如何建立自信？",
    "Explain the nitrogen cycle.",
    "什么是正义？",
    "How do refrigerators work?",
    "请分析一下人口老龄化问题。",
    "What is electricity?",
    "如何有效沟通？",
    "Explain the carbon cycle.",
    "什么是自由？",
    "How do microwaves work?",
    "请评价一下全球化。",
    "What is magnetism?",
    "如何处理压力？",
    "Explain the rock cycle.",
    "什么是真理？",
    "How do lasers work?",
    "请分析一下人工智能伦理。",
    "What is sound?",
    "如何做出正确的决策？",
    "Explain the food chain.",
    "什么是美？",
    "How do satellites work?",
    "请评价一下环境保护。",
    "What is radiation?",
    "如何培养领导力？",
    "Explain the water treatment process.",
    "什么是幸福？",
    "How do solar panels work?",
    "请分析一下数字经济。",
    "What is friction?",
    "如何保持好奇心？",
    "Explain the Krebs cycle.",
    "什么是知识？",
    "How do wind turbines work?",
    "请评价一下社交媒体。",
    "What is buoyancy?",
    "如何克服恐惧？",
    "Explain protein synthesis.",
    "什么是语言？",
    "How do MRI machines work?",
    "请分析一下能源转型。",
    "What is osmosis?",
    "如何找到人生目标？",
    "Explain cellular respiration.",
    "什么是文化？",
    "How do GPS systems work?",
    "请评价一下太空探索。",
    "What is diffusion?",
    "如何建立良好的人际关系？",
    "Explain meiosis and mitosis.",
    "什么是历史？",
    "How do 3D printers work?",
    "请分析一下基因编辑技术。",
    "What is a catalyst?",
    "如何保持学习动力？",
    "Explain the greenhouse effect.",
]

# 注册hook，捕获每层FFN的hidden states
num_layers = len(model.model.layers)
intermediate_size = model.config.intermediate_size

# activation_counts[layer_idx][neuron_idx] = 激活次数
activation_counts = [np.zeros(intermediate_size, dtype=np.int64) for _ in range(num_layers)]
total_tokens = 0

hooks = []

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        global total_tokens
        # output shape: [bsz, seq_len, intermediate_size]
        # SwiGLU: hidden = silu(gate) * up
        # 我们统计 hidden > 0 的位置
        activated = (output > 0).sum(dim=(0, 1))  # [intermediate_size]
        activation_counts[layer_idx] += activated.cpu().numpy()
        total_tokens += output.shape[0] * output.shape[1]
    return hook_fn

# 注册hook到每层MLP的act_fn输出（SwiGLU的hidden state）
for i, layer in enumerate(model.model.layers):
    handle = layer.mlp.act_fn.register_forward_hook(make_hook(i))
    hooks.append(handle)

print(f"Registered {len(hooks)} hooks")
print(f"Model: {num_layers} layers, intermediate_size={intermediate_size}")
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

    if (q_idx + 1) % 10 == 0:
        print(f"  {q_idx + 1}/{NUM_QUESTIONS} done, total_tokens={total_tokens}")

# 移除hooks
for h in hooks:
    h.remove()

print(f"\nTotal tokens processed: {total_tokens}")

# 计算激活频率
frequencies = []
for layer_idx in range(num_layers):
    freq = activation_counts[layer_idx] / max(total_tokens, 1)
    frequencies.append(freq)
    alive = (freq > 0.01).sum()  # >1% 算活跃
    print(f"Layer {layer_idx:2d}: mean_freq={freq.mean():.4f}, max={freq.max():.4f}, "
          f"alive(>1%)={alive}/{intermediate_size} ({100*alive/intermediate_size:.1f}%)")

# 保存分析结果
results = {
    "activation_counts": activation_counts,
    "frequencies": frequencies,
    "total_tokens": total_tokens,
    "num_layers": num_layers,
    "intermediate_size": intermediate_size,
}

with open(os.path.join(OUTPUT_DIR, "activation_analysis.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved analysis to {OUTPUT_DIR}/activation_analysis.pkl")

# 简单统计
all_freqs = np.concatenate(frequencies)
print(f"\nOverall stats:")
print(f"  Mean activation freq: {all_freqs.mean():.4f}")
print(f"  Median: {np.median(all_freqs):.4f}")
print(f"  Max: {all_freqs.max():.4f}")
print(f"  Min: {all_freqs.min():.4f}")
print(f"  Neurons with freq=0: {(all_freqs == 0).sum()} / {len(all_freqs)}")
print(f"  Neurons with freq<0.001: {(all_freqs < 0.001).sum()} / {len(all_freqs)}")
print(f"  Neurons with freq<0.01: {(all_freqs < 0.01).sum()} / {len(all_freqs)}")
print(f"  Neurons with freq<0.05: {(all_freqs < 0.05).sum()} / {len(all_freqs)}")
print(f"  Neurons with freq<0.1: {(all_freqs < 0.1).sum()} / {len(all_freqs)}")
