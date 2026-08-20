import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
from modeling_qwen3_loop import load_model_with_loop

# ========== 配置 ==========
MODEL_PATH = "dynamic_infinite_train_v2/final"
DATA_FILE = "~/data/loop_train_data.npz"
OUTPUT_DIR = "./loop_train_v1"

BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 2000
INITIAL_LR = 1e-4
MIN_LR = 1e-8
WINDOW_SIZE = 5
SAVE_EVERY = 500
REPORT_EVERY = 5

# ========== 加载模型 ==========
print("Loading model with LoopAttention...")
model = load_model_with_loop(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

all_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {all_params/1e6:.1f}M")
print(f"Trainable params: {trainable_params/1e6:.1f}M")
print(f"Loop gate value: {model.model.layers[27].self_attn.loop_gate.item():.4f}")

# ========== 加载预处理好的数据 ==========
print(f"\nLoading preprocessed data from {DATA_FILE}...")
data = np.load(DATA_FILE)
input_ids = torch.from_numpy(data["input_ids"])
attention_mask = torch.from_numpy(data["attention_mask"])
labels = torch.from_numpy(data["labels"])

print(f"Dataset size: {len(input_ids)} samples")
print(f"Sequence length: {input_ids.shape[1]}")

class SimpleDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

dataset = SimpleDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ========== 优化器 ==========
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8)
    print("Using 8-bit AdamW")
except ImportError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8)
    print("Using standard AdamW")

# ========== 训练循环 ==========
print("\nStarting training...")
print(f"Batch size: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}, Effective: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Max steps: {MAX_STEPS}")
print("=" * 60)

model.train()
loss_history = []
best_avg = float('inf')
current_lr = INITIAL_LR
step = 0
accum_loss = 0.0

for epoch in range(100):  # 足够大的epoch数
    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        # 移动数据到GPU
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        accum_loss += loss.item()

        # 梯度累积步
        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

            # 记录loss
            actual_loss = accum_loss
            loss_history.append(actual_loss)
            accum_loss = 0.0

            # 动态无限细LR
            if len(loss_history) >= WINDOW_SIZE:
                window = loss_history[-WINDOW_SIZE:]
                avg = sum(window) / len(window)

                if avg < best_avg:
                    best_avg = avg

                if len(loss_history) >= WINDOW_SIZE + 1:
                    prev_window = loss_history[-(WINDOW_SIZE+1):-1]
                    prev_avg = sum(prev_window) / len(prev_window)
                    improvement = prev_avg - avg

                    if improvement > 0.01:
                        decay = 0.999
                    elif improvement > 0.001:
                        decay = 0.995
                    elif improvement > -0.001:
                        decay = 0.99
                    elif improvement > -0.01:
                        decay = 0.95
                    else:
                        decay = 0.90

                    current_lr = max(current_lr * decay, MIN_LR)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr

            # 报告
            if (step + 1) % REPORT_EVERY == 0:
                gate_val = model.model.layers[27].self_attn.loop_gate.item()
                recent_avg = sum(loss_history[-WINDOW_SIZE:]) / min(WINDOW_SIZE, len(loss_history))
                print(f"[Step {step+1}] loss={actual_loss:.4f} avg={recent_avg:.4f} "
                      f"lr={current_lr:.2e} best={best_avg:.4f} gate={gate_val:.4f}")

            # 保存
            if (step + 1) % SAVE_EVERY == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  -> Saved checkpoint to {save_path}")

            # 停止条件
            if current_lr < MIN_LR:
                print(f"\nLR reached {MIN_LR:.0e}. Stopping.")
                step = MAX_STEPS
                break

        step += 1

    if step >= MAX_STEPS:
        break

# 保存最终模型
final_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_path, exist_ok=True)
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"\nTraining complete. Saved to {final_path}")
print(f"Final loop_gate: {model.model.layers[27].self_attn.loop_gate.item():.4f}")
