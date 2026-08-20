import json
import os
import numpy as np
from transformers import AutoTokenizer

MODEL_PATH = "dynamic_infinite_train_v2/final"
MAX_LENGTH = 512
MAX_CHAR_LEN = 10000
DATA_FILES = [
    "/tmp/training-data/sft_final_v3_p1.jsonl",
    "/tmp/training-data/sft_final_v3_p2.jsonl",
]
ANTILOOP_FILE = "/tmp/training-data/sft_antiloop.jsonl"
OUTPUT_FILE = "/tmp/training-data/loop_train_data.npz"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

all_texts = []

# 加载原始数据（过滤超长）
for fpath in DATA_FILES:
    if not os.path.exists(fpath):
        continue
    skipped = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            txt = tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
            if len(txt) <= MAX_CHAR_LEN:
                all_texts.append(txt)
            else:
                skipped += 1
    print(f"  {fpath}: loaded, skipped {skipped}")

# 加载反例数据（20%子集，过滤超长）
if os.path.exists(ANTILOOP_FILE):
    skipped = 0
    with open(ANTILOOP_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i % 5 != 0:
                continue
            d = json.loads(line)
            txt = tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
            if len(txt) <= MAX_CHAR_LEN:
                all_texts.append(txt)
            else:
                skipped += 1
    print(f"  Antiloop: loaded, skipped {skipped}")

print(f"\nTotal samples: {len(all_texts)}")
print(f"Max char length: {max(len(t) for t in all_texts)}")

# 逐条tokenize，避免dataset.map()的缓存问题
print("Tokenizing...")
input_ids_list = []
attention_mask_list = []
labels_list = []

for i, text in enumerate(all_texts):
    tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids_list.append(tokens["input_ids"])
    attention_mask_list.append(tokens["attention_mask"])
    labels_list.append(tokens["input_ids"])  # labels = input_ids for CLM
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(all_texts)} done")

# 转成numpy数组
input_ids = np.array(input_ids_list, dtype=np.int32)
attention_mask = np.array(attention_mask_list, dtype=np.int32)
labels = np.array(labels_list, dtype=np.int32)

print(f"\nShapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
print(f"Saving to {OUTPUT_FILE}...")
np.savez(OUTPUT_FILE, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print("Done!")
