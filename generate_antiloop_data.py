import json
import random
import os
import re

def split_sentences(text):
    sentences = re.split(r'([。！？\.!?])', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        s = s.strip()
        if s:
            result.append(s)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result

BREAK_MARKERS = [
    "等等，我刚才在重复自己。让我重新组织一下：\n\n",
    "停一下，我意识到自己在循环。换个角度来说：\n\n",
    "抱歉，我在重复前面的内容。让我用另一种方式表达：\n\n",
    "检查：前面内容已重复。修正如下：\n\n",
    "注意：我陷入了重复模式。重新整理思路：\n\n",
    "重复警告。切换视角重新阐述：\n\n",
]

def build_antiloop_sample(original_messages, max_assistant_len=5000):
    """
    只从assistant回复较短的样本构造反例，避免生成超长文本。
    """
    messages = [dict(m) for m in original_messages]

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            original = messages[i]["content"]

            # 过滤：assistant回复太长的不构造反例
            if len(original) > max_assistant_len:
                return None

            sentences = split_sentences(original)

            if len(sentences) < 3:
                return None

            prefix = "\n\n".join(sentences[:2])
            loop_part = "\n\n".join([prefix] * 3)

            marker = random.choice(BREAK_MARKERS)
            new_content = loop_part + "\n\n" + marker + original

            # 最终长度检查
            if len(new_content) > 15000:
                return None

            messages[i]["content"] = new_content
            return {"messages": messages}

    return None

def main():
    input_files = [
        "/tmp/training-data/sft_final_v3_p1.jsonl",
        "/tmp/training-data/sft_final_v3_p2.jsonl"
    ]
    output_file = "/tmp/training-data/sft_antiloop.jsonl"

    all_original = []
    for fpath in input_files:
        if not os.path.exists(fpath):
            print(f"Skip missing: {fpath}")
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_original.append(json.loads(line))

    print(f"Loaded {len(all_original)} original samples")

    antiloop_samples = []
    for item in all_original:
        antiloop = build_antiloop_sample(item["messages"], max_assistant_len=5000)
        if antiloop:
            antiloop_samples.append(antiloop)

    print(f"Generated {len(antiloop_samples)} antiloop samples (filtered long ones)")

    with open(output_file, 'w', encoding='utf-8') as f:
        for s in antiloop_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved to {output_file}")

    # 统计
    lengths = []
    for s in antiloop_samples:
        for m in s["messages"]:
            if m.get("role") == "assistant":
                lengths.append(len(m["content"]))
    if lengths:
        print(f"Assistant content lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")

if __name__ == "__main__":
    main()
