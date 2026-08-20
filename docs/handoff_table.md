# 实验转接表：WSL → AutoDL 迁移
**生成时间**：2026-08-21 02:11  
**当前平台**：WSL (frpc 26026) → 目标：AutoDL  
**原因**：WSL 内存/稳定性瓶颈，AutoDL 提供更稳定的 GPU 训练环境

---

## 一、当前进行中的任务（精确状态）

### 任务 A：LoopAttention 训练（未完成）
| 项目 | 详情 |
|------|------|
| 状态 | ❌ WSL 上未跑通，待 AutoDL 重新执行 |
| 核心文件 | `~/modeling_qwen3_loop.py`（Attention 修改模块） |
| | `~/train_loop_manual.py`（手写 DataLoader 版，推荐） |
| | `~/train_with_loop.py`（Trainer API 版，有 dataset.map() OOM 问题） |
| 训练数据 | `/tmp/training-data/sft_final_v3_p1.jsonl` (~3000条) |
| | `/tmp/training-data/sft_final_v3_p2.jsonl` (~3000条) |
| | `/tmp/training-data/sft_antiloop.jsonl` (~5400条，已过滤超长) |
| 基座模型 | `~/dynamic_infinite_train_v2/final/` (Qwen3-0.6B SFT后) |
| 关键参数 | `loop_gate` 初始值 0.0，只改第 27 层，eager 模式 |
| 阻塞原因 | WSL dataset.map() 内存峰值导致进程消失 |
| **AutoDL 建议** | 直接用 `train_loop_manual.py`（绕过 datasets 库） |

### 任务 B：FFN 激活频率分析（运行中）
| 项目 | 详情 |
|------|------|
| 状态 | 🔄 **正在运行** PID 96905（frpc 26026 服务器上） |
| 脚本 | `~/analyze_quick.py`（20 问题快速版） |
| | `~/analyze_activation.py`（100 问题完整版，备用） |
| 日志 | `~/analyze_quick_nohup.log` |
| 预期输出 | `~/pruning_analysis/activation_analysis.pkl` |
| 运行时间 | 20 问题约 1-2 分钟，100 问题约 10-15 分钟 |
| **AutoDL 建议** | 到 AutoDL 后重新跑 `analyze_activation.py`（100问题） |

### 任务 C：结构化剪枝（待执行）
| 项目 | 详情 |
|------|------|
| 状态 | ⏳ 等待任务 B 完成 |
| 脚本 | `~/prune_ffn.py` |
| 输入 | `~/pruning_analysis/activation_analysis.pkl` |
| 输出 | `~/pruned_model_50pct/` (~0.3B) |
| 剪枝比例 | 50%（保守），后续可尝试 70-90% |
| **AutoDL 建议** | 分析完成后直接执行 |

### 任务 D：对比测试（待执行）
| 项目 | 详情 |
|------|------|
| 状态 | ⏳ 等待任务 C 完成 |
| 脚本 | `~/test_pruned.py` |
| 对比对象 | 原始 `dynamic_infinite_train_v2/final` vs 剪枝后 `pruned_model_50pct` |
| 输出 | `~/pruning_analysis/pruned_comparison.json` |
| **AutoDL 建议** | 剪枝完成后执行 |

---

## 二、全部文件清单（WSL 服务器 43.160.201.202:26026）

### 核心脚本（~/ 目录）
```
~/modeling_qwen3_loop.py          # LoopAttention 模块
~/train_with_loop.py              # Trainer API 版训练（有OOM问题）
~/train_loop_manual.py            # 手写 DataLoader 版训练（推荐）
~/analyze_activation.py           # 100问题 FFN 激活分析
~/analyze_quick.py                # 20问题快速分析
~/prune_ffn.py                    # 结构化剪枝 50%
~/test_pruned.py                  # 原始 vs 剪枝对比测试
~/generate_antiloop_data.py       # 反例数据构造
~/diag_inference.py               # 推理诊断脚本
~/loop_train_nohup.log            # 旧训练日志（失败记录）
~/loop_manual_nohup.log           # 手动训练日志
~/analyze_nohup.log               # 100问题分析日志
~/analyze_quick_nohup.log         # 20问题分析日志（当前运行中）
```

### 数据文件
```
/tmp/training-data/sft_final_v3_p1.jsonl      # 原始SFT数据前半
/tmp/training-data/sft_final_v3_p2.jsonl      # 原始SFT数据后半
/tmp/training-data/sft_antiloop.jsonl         # 反例数据（已过滤超长）
/tmp/training-data/loop_train_data.npz        # 预处理后的numpy数据（如有）
```

### 模型文件
```
~/dynamic_infinite_train_v2/final/            # SFT基座模型（~1.2GB）
~/dynamic_infinite_train_v2/checkpoint-500/   # 中间检查点
~/dynamic_infinite_train_v2/checkpoint-1000/  # 中间检查点
~/loop_train_v1/final/                        # LoopAttention训练输出（如有）
~/pruned_model_50pct/                         # 剪枝后模型（待生成）
~/pruning_analysis/                           # 分析结果目录
```

### 论文
```
本地：/mnt/agents/output/Qwen3_0.6B_微调循环抑制与剪枝实验报告.md
待上传桦树工坊：/home/ubuntu/01_理论体系/论文集/6域 VI：AI、认知与人机交互/
```

---

## 三、AutoDL 迁移执行清单

### Step 0：环境准备
```bash
# 1. 创建 AutoDL 实例（推荐 RTX 3090/4090 24GB 或 A100 40GB）
# 2. 登录后安装依赖
pip install transformers==4.48.0 datasets torch accelerate bitsandbytes -q

# 3. 从 WSL 服务器下载所有文件
# 方法A：scp（如果WSL有公网IP）
scp -P 26026 ubuntu@43.160.201.202:~/modeling_qwen3_loop.py .
scp -P 26026 ubuntu@43.160.201.202:~/train_loop_manual.py .
scp -P 26026 ubuntu@43.160.201.202:~/analyze_activation.py .
scp -P 26026 ubuntu@43.160.201.202:~/prune_ffn.py .
scp -P 26026 ubuntu@43.160.201.202:~/test_pruned.py .
scp -P 26026 -r ubuntu@43.160.201.202:~/dynamic_infinite_train_v2/final/ ./base_model/
scp -P 26026 ubuntu@43.160.201.202:/tmp/training-data/*.jsonl ./data/

# 方法B：如果scp不稳定，先打包再传
ssh -p 26026 ubuntu@43.160.201.202 "cd ~ && tar czf experiment.tar.gz modeling_qwen3_loop.py train_loop_manual.py analyze_activation.py prune_ffn.py test_pruned.py generate_antiloop_data.py dynamic_infinite_train_v2/final/"
scp -P 26026 ubuntu@43.160.201.202:~/experiment.tar.gz .
tar xzf experiment.tar.gz
```

### Step 1：重新生成反例数据（可选，如数据已传则跳过）
```bash
python3 generate_antiloop_data.py
# 输出：/tmp/training-data/sft_antiloop.jsonl
```

### Step 2：LoopAttention 训练（优先）
```bash
# 推荐用 train_loop_manual.py（绕过 datasets 库）
nohup python3 -u train_loop_manual.py > loop_train_autodl.log 2>&1 &
tail -f loop_train_autodl.log

# 预期：2000步，约30-60分钟（取决于GPU）
# 关键观察：loop_gate 值应从 0.0 逐渐变为正数
```

### Step 3：FFN 激活频率分析
```bash
# 100问题完整版
nohup python3 -u analyze_activation.py > analyze_autodl.log 2>&1 &
tail -f analyze_autodl.log

# 预期：10-15分钟，输出 pruning_analysis/activation_analysis.pkl
```

### Step 4：结构化剪枝
```bash
python3 prune_ffn.py
# 输出：pruned_model_50pct/
# 预期：模型从 ~0.6B → ~0.3B
```

### Step 5：对比测试
```bash
python3 test_pruned.py
# 输出：pruning_analysis/pruned_comparison.json
# 对比：原始 vs 剪枝后的速度、质量
```

### Step 6：激进剪枝实验（可选）
```bash
# 修改 prune_ffn.py 中的 PRUNE_RATIO = 0.70 或 0.90
# 重复 Step 4-5，测试 70%/90% 剪枝的可行性
```

---

## 四、已知陷阱与避坑指南

| 陷阱 | 现象 | 修复 |
|------|------|------|
| `past_key_value` 单复数 | `TypeError` 或模型加载失败 | 子类 `forward` 必须用 `past_key_value`（单数），与父类完全一致 |
| `dataset.map()` OOM | 进程在 Map 阶段消失，无 traceback | 过滤超长样本（>10000字符），或用 `train_loop_manual.py` 绕过 |
| Python stdout 缓冲 | 日志长时间不更新，误以为进程卡死 | 启动加 `PYTHONUNBUFFERED=1`，或 `print(..., flush=True)` |
| WSL 无显存隔离 | 加载大模型时系统蓝屏 | AutoDL 无此问题，但仍需监控 `nvidia-smi` |
| frpc 26026 断开 | SSH 会话随机中断 | AutoDL 有稳定公网 IP，无需 frp |
| 反例数据超长 | 75万字符样本导致内存爆炸 | 构造时过滤 `len(original) > 5000` |
| 8-bit AdamW 缺失 | 回退到标准 AdamW，显存占用更高 | `pip install bitsandbytes` |
| `attn_implementation` | LoopAttention 必须用 `eager` | `sdpa`/`flash_attention_2` 拿不到显式 `attn_weights` |

---

## 五、关键配置速查

### WSL 服务器（源）
- **IP**: 43.160.201.202
- **Port**: 26026 (frpc)
- **User**: ubuntu
- **Password**: !Suk416520
- **模型路径**: `~/dynamic_infinite_train_v2/final/`
- **数据路径**: `/tmp/training-data/`

### 桦树工坊（论文上传目标）
- **IP**: 43.160.201.202
- **Port**: 22 (SSH直连)
- **User**: ubuntu
- **Password**: !Suk416520
- **论文路径**: `/home/ubuntu/01_理论体系/论文集/6域 VI：AI、认知与人机交互/`
- **上传命令**:
```bash
scp -P 22 Qwen3_0.6B_微调循环抑制与剪枝实验报告.md ubuntu@43.160.201.202:/home/ubuntu/01_理论体系/论文集/6域\ VI：AI、认知与人机交互/
```

### AutoDL（目标）
- **待创建**：推荐 RTX 3090 24GB 或更高
- **系统镜像**：PyTorch 2.4 + CUDA 12.1
- **需要安装**: `transformers`, `datasets`, `accelerate`, `bitsandbytes`

---

## 六、论文上传（立即执行）

论文文件在本地：`/mnt/agents/output/Qwen3_0.6B_微调循环抑制与剪枝实验报告.md`

需要上传到桦树工坊 22 端口服务器的论文集目录。

---

## 七、验证检查点

迁移到 AutoDL 后，按以下顺序验证：

1. [ ] `python3 -c "import torch; print(torch.cuda.is_available())"` → True
2. [ ] `python3 diag_inference.py` → 模型能正常推理
3. [ ] `python3 analyze_activation.py` → 100问题分析完成，生成 pkl
4. [ ] `python3 prune_ffn.py` → 生成 pruned_model_50pct/
5. [ ] `python3 test_pruned.py` → 有对比结果 JSON
6. [ ] `python3 train_loop_manual.py` → LoopAttention 训练完成，loop_gate > 0

---

*本转接表由人机协同实验组生成，确保实验连续性。*
