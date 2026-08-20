import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, TrainingArguments, TrainerCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import math

from modeling_qwen3_loop import load_model_with_loop

# ========== 配置 ==========
MODEL_PATH = "dynamic_infinite_train_v2/final"
OUTPUT_DIR = "./loop_train_v1"
DATA_FILES = [
    "/tmp/training-data/sft_final_v3_p1.jsonl",
    "/tmp/training-data/sft_final_v3_p2.jsonl",
]
ANTILOOP_FILE = "/tmp/training-data/sft_antiloop.jsonl"

MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 2000
INITIAL_LR = 1e-4
MIN_LR = 1e-8
WINDOW_SIZE = 5
MAX_CHAR_LEN = 10000

# ========== 加载模型 ==========
print("Loading model with LoopAttention (last layer)...")
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

# ========== 数据加载（过滤超长样本）==========
def load_and_filter(fpath, tokenizer, max_char=MAX_CHAR_LEN):
    data = []
    skipped = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            txt = tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
            if len(txt) <= max_char:
                data.append(txt)
            else:
                skipped += 1
    print(f"  {fpath}: loaded {len(data)}, skipped {skipped} (>{max_char} chars)")
    return data

all_texts = []
for fpath in DATA_FILES:
    if os.path.exists(fpath):
        all_texts.extend(load_and_filter(fpath, tokenizer))

antiloop_texts = []
if os.path.exists(ANTILOOP_FILE):
    skipped = 0
    with open(ANTILOOP_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i % 5 != 0:
                continue
            d = json.loads(line)
            txt = tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
            if len(txt) <= MAX_CHAR_LEN:
                antiloop_texts.append(txt)
            else:
                skipped += 1
    print(f"  Antiloop: loaded {len(antiloop_texts)}, skipped {skipped} (>{MAX_CHAR_LEN} chars)")

all_texts.extend(antiloop_texts)

print(f"\nTotal training samples: {len(all_texts)}")
print(f"  Original: {len(all_texts) - len(antiloop_texts)}")
print(f"  Antiloop (20% subset): {len(antiloop_texts)}")

max_len = max(len(t) for t in all_texts)
print(f"Max sample length: {max_len} chars")

def encode(text):
    tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_dict({"text": all_texts})
dataset = dataset.map(lambda x: encode(x["text"]), remove_columns=["text"], num_proc=1)

# ========== Callback ==========
class DynamicInfiniteCallback(TrainerCallback):
    def __init__(self, optimizer_ref):
        self.optimizer_ref = optimizer_ref
        self.loss_history = []
        self.best_avg = float('inf')
        self.lr = INITIAL_LR
        self.decay_factors = []
        self.report_interval = 5
        self.last_report_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        opt = self.optimizer_ref[0]
        if state.global_step <= 2:
            warmup_lr = INITIAL_LR * (state.global_step / 2)
            for pg in opt.param_groups:
                pg['lr'] = warmup_lr
            if state.global_step == 1:
                print(f"[Step {state.global_step}] Warmup LR: {warmup_lr:.2e}")
        else:
            for pg in opt.param_groups:
                pg['lr'] = self.lr

    def on_step_end(self, args, state, control, **kwargs):
        logs = state.log_history
        if not logs or "loss" not in logs[-1]:
            return control

        loss = logs[-1]["loss"]
        self.loss_history.append(loss)

        if len(self.loss_history) >= WINDOW_SIZE:
            window = self.loss_history[-WINDOW_SIZE:]
            avg = sum(window) / len(window)

            if avg < self.best_avg:
                self.best_avg = avg

            if len(self.loss_history) >= WINDOW_SIZE + 1:
                prev_window = self.loss_history[-(WINDOW_SIZE+1):-1]
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

                self.lr = max(self.lr * decay, MIN_LR)
                self.decay_factors.append(decay)

            if state.global_step - self.last_report_step >= self.report_interval:
                self.last_report_step = state.global_step
                gate_val = kwargs['model'].model.layers[27].self_attn.loop_gate.item()
                print(f"[Step {state.global_step}] loss={loss:.3f} avg={avg:.4f} "
                      f"lr={self.lr:.2e} best={self.best_avg:.4f} "
                      f"loop_gate={gate_val:.4f}")

        if self.lr < MIN_LR:
            print(f"[Step {state.global_step}] LR reached {MIN_LR:.0e}. Stopping.")
            control.should_training_stop = True

        return control

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=INITIAL_LR,
    warmup_steps=0,
    logging_steps=1,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    disable_tqdm=True,
    report_to="none",
    dataloader_num_workers=0,
)

try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8)
    print("Using 8-bit AdamW")
except ImportError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8)
    print("Using standard AdamW")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

optimizer_ref = [optimizer]
callback = DynamicInfiniteCallback(optimizer_ref)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[callback],
)

print("\nStarting training with LoopAttention...")
print(f"Data: {len(all_texts)} samples")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = effective {BATCH_SIZE * GRAD_ACCUM}")
print(f"Max steps: {MAX_STEPS}")
print("="*60)

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print(f"\nTraining complete. Saved to {OUTPUT_DIR}/final")
print(f"Final loop_gate: {model.model.layers[27].self_attn.loop_gate.item():.4f}")
