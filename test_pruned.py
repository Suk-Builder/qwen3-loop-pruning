import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

ORIGINAL_PATH = "dynamic_infinite_train_v2/final"
PRUNED_PATH = "./pruned_model_50pct"
OUTPUT_FILE = "./pruning_analysis/pruned_comparison.json"

TEST_QUESTIONS = [
    "什么是量子力学？",
    "如何学习编程？",
    "人工智能会取代人类吗？",
    "请解释区块链技术。",
    "什么是民主制度？",
    "请分析当前经济形势。",
    "如何保持健康的生活方式？",
    "什么是马克思主义？",
    "如何写好一篇论文？",
    "请评价一下莎士比亚。",
]

def load_and_test(model_path, name, questions, max_new_tokens=256):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {model_path}")

    if not os.path.exists(model_path):
        print(f"SKIP: {model_path} does not exist")
        return None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.1f}M")

    results = []
    total_time = 0
    total_tokens = 0

    for q_idx, question in enumerate(questions):
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        num_new = len(outputs[0]) - len(inputs.input_ids[0])

        total_time += elapsed
        total_tokens += num_new

        results.append({
            "question": question,
            "response": response,
            "tokens": num_new,
            "time": elapsed,
        })

        print(f"  Q{q_idx+1}: {num_new} tokens in {elapsed:.2f}s")

    avg_time = total_time / len(questions)
    avg_tokens = total_tokens / len(questions)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    print(f"\nSummary: avg {avg_tokens:.0f} tokens/q, {avg_time:.2f}s/q, {tokens_per_sec:.1f} tok/s")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "params_m": params / 1e6,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "results": results,
    }

# 测试原始模型
original_result = load_and_test(ORIGINAL_PATH, "Original (0.6B)", TEST_QUESTIONS)

# 测试剪枝模型
pruned_result = load_and_test(PRUNED_PATH, "Pruned (~0.3B)", TEST_QUESTIONS)

# 保存对比
comparison = {
    "original": original_result,
    "pruned": pruned_result,
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print(f"\nComparison saved to {OUTPUT_FILE}")

if original_result and pruned_result:
    print(f"\n{'='*60}")
    print("FINAL COMPARISON:")
    print(f"  Original:  {original_result['params_m']:.1f}M params, {original_result['tokens_per_sec']:.1f} tok/s")
    print(f"  Pruned:    {pruned_result['params_m']:.1f}M params, {pruned_result['tokens_per_sec']:.1f} tok/s")
    print(f"  Speedup:   {pruned_result['tokens_per_sec']/original_result['tokens_per_sec']:.2f}x")
    print(f"  Size reduction: {(1 - pruned_result['params_m']/original_result['params_m'])*100:.1f}%")
