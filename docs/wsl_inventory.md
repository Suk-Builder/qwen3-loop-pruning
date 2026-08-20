# WSL 服务器完整环境盘点
**服务器**: 43.160.201.202:26026 (frpc)  
**用户**: ubuntu  
**生成时间**: 2026-08-21 02:52  
**用途**: Qwen3-0.6B 微调、循环抑制、结构化剪枝实验环境

---

## 一、硬件与系统环境

| 项目 | 详情 |
|------|------|
| OS | Linux Baihua 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC |
| CPU | AMD Ryzen 5 5600 6-Core Processor (12 线程) |
| 内存 | 27GB 物理，25GB 可用，7GB Swap |
| GPU | NVIDIA GeForce RTX 3060 12GB |
| 驱动 | 591.86 |
| 显存 | 12288 MiB 总，11482 MiB 空闲，629 MiB 已用 |
| 磁盘 | /dev/sdd 1007G，已用 579G，可用 378G (61%) |
| Python | 3.10.12 |
| Shell | zsh (oh-my-zsh) |

---

## 二、Python 关键包环境

| 包名 | 版本 | 用途 |
|------|------|------|
| transformers | 5.14.1 | LLM 训练/推理 |
| torch | 2.5.1 | 深度学习框架 |
| datasets | 5.0.1 | 数据加载 |
| accelerate | 1.14.0 | 分布式训练 |
| bitsandbytes | 0.50.0 | 8-bit 优化器 |
| peft | 0.20.0 | LoRA/QLoRA |
| trl | 1.10.0 | RLHF/SFT |
| tokenizers | 0.22.2 | 分词 |
| sentencepiece | 0.2.2 | 子词分词 |
| numpy | 2.2.6 | 数值计算 |
| scipy | 1.15.3 | 科学计算 |
| huggingface_hub | 1.27.0 | 模型下载 |
| pytorch-lightning | 2.6.5 | 训练框架 |
| torchvision | 0.20.1 | 视觉 |
| onnx2torch-py313 | 1.6.0 | ONNX 转换 |
| open_clip_torch | 3.3.3 | CLIP |
| rotary-embedding-torch | 0.6.5 | RoPE |
| torchdiffeq | 0.2.5 | 神经 ODE |
| torchmetrics | 1.9.0 | 指标 |
| torchsde | 0.2.6 | 随机微分方程 |

---

## 三、家目录文件总览（按类别）

### 3.1 本次实验核心脚本（~80 个 Python 文件）

**训练相关**:
- `train_0.5b_full.py`, `train_0.5b_resume.py`, `train_0.5b_safe.py`, `train_0.5b_test.py`, `train_0.5b_v2.py`
- `train_0.6b_full.py`, `train_06b_focused.py`, `train_06b_simple.py`, `train_06b_standard.py`
- `train_3b_v2.py`, `train_baihua_3b.py`, `train_final.py`, `train_self_contained.py`
- `train_stateful_0.6b.py`, `train_with_loop.py`, `train_loop_manual.py`
- `dynamic_infinite_train.py`, `dynamic_infinite_train_v2.py`
- `infinite_fine_train.py`, `adaptive_train_v3.py`, `adaptive_train_v3_1.py`
- `adaptive_machining.py`, `adaptive_machining_v2.py`, `machining_train.py`, `machining_train_v2.py`
- `iterative_15x3.py`, `three_stage.py`, `two_stage_35.py`
- `short_coarse_scan.py`, `lr_scan.py`, `lr_scan_v2.py`, `lr_scan_50step.py`
- `grid_search.py`

**数据相关**:
- `generate_antiloop_data.py`, `preprocess_loop_data.py`
- `extract_negative.py`, `extract_current.py`
- `append_auto_mined.py`, `save_auto_mined.py`
- `bulk_synthetic_550b.py`, `fixed_synthetic_550b.py`, `flash_synthetic.py`, `synthetic_550b.py`, `synthetic_550b_batch.py`
- `gen_questions.py`, `gen_5000_questions.py`
- `reformat_data.py`, `merge_data.py`
- `data_pipeline.py`

**模型/Attention 相关**:
- `modeling_qwen3_loop.py` — LoopAttention 模块
- `lightweight_stateful_attention.py` — 轻量级状态 Attention
- `prune_ffn.py` — FFN 结构化剪枝
- `prune_0.6b.py` — 0.6B 模型剪枝
- `analyze_activation.py` — FFN 激活频率分析（100 问题）
- `analyze_quick.py` — 快速分析（20 问题）
- `test_pruned.py` — 剪枝后对比测试
- `diag_inference.py` — 推理诊断
- `test_chat.py`, `test_long_generation.py`, `test_long_prompt.py`, `test_long_stream.py`
- `test_multi_domain.py`, `test_extreme.py`, `test_2000token.py`, `test_2000_bg.py`
- `test_baihua_bg.py`, `test_baihua_streaming.py`, `test_baihua_think.py`
- `test_agent.py`, `test_modao_extract.py`, `test_0.5b_simple.py`
- `min_verify_loop_break.py` — 循环打断最小验证

**Kimi/数据导出**:
- `kimi_export_api.py`, `kimi_export_direct.py`, `kimi_export_final.py`, `kimi_export_fix.py`
- `kimi_export_headless.py`, `kimi_export_playwright.py`, `kimi_export_v2.py`
- `kimi_gui_export.py`, `kimi_browser_use.py`, `kimi_api_export.py`, `kimi_batch_export.py`

**GUI/API**:
- `gui_api.py`, `gui_test.py`, `gui_robust.py`
- `gpu_watchdog.py`
- `probe_a11y.py`, `probe_locator.py`

**其他**:
- `ab_test.py` — A/B 测试
- `download_model.py`, `dl_model.py`
- `uvr_web.py` — 音频分离
- `limit_test_2000.py`

### 3.2 Shell 脚本

- `download_uncensored.sh` — 下载无审查模型
- `final_fix.sh` — 最终修复
- `launch_train.sh` — 启动训练
- `model_download.sh` — 模型下载

### 3.3 日志文件（~20 个）

| 日志 | 大小 | 说明 |
|------|------|------|
| `gui_api.log` | 518K | GUI API 日志 |
| `adaptive_train_v3_1_nohup.log` | 67K | 自适应训练 v3.1 |
| `dynamic_infinite_train_nohup.log` | 27K | 动态无限细训练 |
| `dynamic_infinite_train_v2_nohup.log` | 22K | 动态无限细训练 v2 |
| `grid_search.log` | 188K | 网格搜索 |
| `iterative_15x3.log` | 17K | 迭代 15×3 |
| `infinite_fine_train_nohup.log` | 16K | 无限细训练 |
| `ab_test.log` | 15K | A/B 测试 |
| `adaptive.log` | 3.4K | 自适应训练 |
| `adaptive_v2.log` | 2.4K | 自适应 v2 |
| `analyze_nohup.log` | 2.8K | 激活分析（100 问题，失败） |
| `analyze_quick_nohup.log` | 4.7K | 快速分析（20 问题，运行中） |
| `loop_manual_nohup.log` | — | 手动训练日志 |
| `loop_train_nohup.log` | — | 旧训练日志 |
| `frpc.log` | 0 | frpc 日志（空） |
| `frpc.20260821-000032.log` | 1.5K | frpc 历史日志 |
| `gpu_watchdog.log` | 2K | GPU 监控 |
| `gui_robust.log` | 278 | GUI 测试 |
| `gui_test.log` | 186 | GUI 测试 |
| `kimi_api_export.log` | 757 | Kimi 导出 |

### 3.4 配置文件

- `.bashrc` — Bash 配置
- `.zshrc` — Zsh 配置（oh-my-zsh）
- `.zshenv` — Zsh 环境
- `.gitconfig` — Git 配置
- `.git-credentials` — Git 凭据
- `.condarc` — Conda 配置
- `frpc.toml` / `frpc.ini` — frp 内网穿透配置
- `ENV-VARS` — 环境变量文件
- `AI-README` — AI 说明
- `README.md` — 项目说明
- `DOWNLOAD_HANDBOOK.md` — 下载手册
- `_INDEX` — 索引

### 3.5 JSON 数据文件

- `grid_search_results.json` — 网格搜索结果
- `lr_scan2_results.json` — 学习率扫描结果

---

## 四、训练输出目录（~30 个，各 ~4.5G）

| 目录 | 大小 | 说明 |
|------|------|------|
| `dynamic_infinite_train/` | ~4.5G | 动态无限细训练输出 |
| `dynamic_infinite_train_v2/` | ~4.5G | 动态无限细训练 v2（含 final/） |
| `infinite_fine_train/` | ~4.5G | 无限细训练 |
| `adaptive_train_v3_1/` | ~4.5G | 自适应训练 v3.1 |
| `adaptive_v2/` | ~4.5G | 自适应 v2 |
| `adaptive_machining/` | ~4.5G | 自适应加工 |
| `machining_train_v2/` | ~4.5G | 加工训练 v2 |
| `iter15x3_round1/` | ~4.5G | 迭代 15×3 第一轮 |
| `iter15x3_round2/` | ~4.5G | 迭代 15×3 第二轮 |
| `iter15x3_round3/` | ~4.5G | 迭代 15×3 第三轮 |
| `linear_15step/` | ~4.5G | 线性 15 步 |
| `three_stage_s1/` | ~4.5G | 三阶段 s1 |
| `three_stage_s2/` | ~4.5G | 三阶段 s2 |
| `three_stage_s3/` | ~4.5G | 三阶段 s3 |
| `two35_s1/` | ~4.5G | 两阶段 35 s1 |
| `two35_s2/` | ~4.5G | 两阶段 35 s2 |
| `short_coarse_5/` | ~4.5G | 短粗调 5 |
| `short_coarse_10/` | ~4.5G | 短粗调 10 |
| `short_coarse_15/` | ~4.5G | 短粗调 15 |
| `short_fine_5/` | ~3.4G | 短精调 5 |
| `short_fine_10/` | ~3.4G | 短精调 10 |
| `short_fine_15/` | ~3.4G | 短精调 15 |
| `pp-sft-0.5b/` | ~6.7G | 0.5B SFT |
| `pp-sft-0.5b-safe/` | ~22G | 0.5B 安全版 |
| `pp-sft-0.5b-fast/` | ~4.5G | 0.5B 快速 |
| `pp-sft-0.5b-fast-stage1/` | ~4.5G | 0.5B 快速 stage1 |
| `pp-sft-0.5b-fast2/` | ~4.5G | 0.5B 快速 2 |
| `pp-sft-0.5b-fast2-stage1/` | ~4.5G | 0.5B 快速 2 stage1 |
| `grid_s20_coarse/` | ~4.5G | 网格 s20 粗调 |
| `grid_s20_f30/` | ~4.5G | 网格 s20 f30 |
| `grid_s20_f40/` | ~4.5G | 网格 s20 f40 |
| `grid_s20_f50/` | ~4.5G | 网格 s20 f50 |
| `grid_s30_coarse/` | ~4.5G | 网格 s30 粗调 |
| `grid_s30_f30/` | ~4.5G | 网格 s30 f30 |
| `grid_s30_f40/` | ~4.5G | 网格 s30 f40 |
| `grid_s30_f50/` | ~4.5G | 网格 s30 f50 |
| `grid_s40_coarse/` | ~4.5G | 网格 s40 粗调 |
| `grid_s40_f30/` | ~4.5G | 网格 s40 f30 |
| `grid_s40_f40/` | ~4.5G | 网格 s40 f40 |
| `grid_s40_f50/` | ~4.5G | 网格 s40 f50 |
| `lr_scan_3e-04/` | ~3.4G | 学习率扫描 |
| `ab_stage1/` | ~4.5G | A/B stage1 |
| `ab_group_a/` | — | A/B 组 A |

---

## 五、模型文件

| 路径 | 大小 | 说明 |
|------|------|------|
| `~/models/` | 85G | 所有模型总目录 |
| `~/dynamic_infinite_train_v2/final/` | 1.2G | Qwen3-0.6B SFT 基座（本次实验起点） |
| `~/miniconda3/` | 22G | Conda 环境 |

---

## 六、训练数据

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `/tmp/training-data/sft_final_v3_p1.jsonl` | 21M | 3000 | SFT 数据前半 |
| `/tmp/training-data/sft_final_v3_p2.jsonl` | 23M | 3000 | SFT 数据后半 |
| `/tmp/training-data/sft_final_v3_p3.jsonl` | 21M | 3000 | SFT 数据 p3 |
| `/tmp/training-data/sft_final_v3_p4.jsonl` | 20M | 3000 | SFT 数据 p4 |
| `/tmp/training-data/sft_final_v2_p1.jsonl` | 93M | 5000 | SFT v2 前半 |
| `/tmp/training-data/sft_final_v2_p2.jsonl` | 89M | 5000 | SFT v2 后半 |
| `/tmp/training-data/sft_antiloop.jsonl` | 33M | 5415 | 反例数据（已过滤） |
| `/tmp/training-data/deepseek_extracted.jsonl` | 35M | 6027 | DeepSeek 提取数据 |
| `/tmp/training-data/deepseek_conversations_raw.json` | 43M | — | DeepSeek 原始对话 |
| `/tmp/training-data/loop_train_data.npz` | 41M | — | 预处理 numpy 数据 |
| `/tmp/training-data/README.md` | 556 | — | 数据说明 |

**总计**: ~33442 条对话，~413MB

---

## 七、项目体系结构（01-07）

```
~/01-Principia/          (4.3G)
  ├── BDI/               # BDI 架构
  ├── 数学工具/          # CG 数学工具
  ├── 飓风预测/          # 飓风预测项目
  └── The-Law-of-Circles/# 圆律理论体系

~/02-Brain/              # 大脑/认知体系
  ├── CrackNeural/       # 神经网络破解
  ├── baihua/            # 白话项目
  ├── cells/             # 细胞级模拟
  ├── brain_v9/          # 大脑 v9
  ├── Brain/             # 大脑核心
  ├── hippocampus/       # 海马体
  └── hier_training_bundle/ (软链: ~/hier-training)

~/03-Agents/             # Agent 体系
  ├── BSEM/              # BSEM Agent
  ├── CCAGI/             # CCAGI
  └── cg-agent/          # CG Agent

~/04-Products/           # 产品
  ├── baihua-app/        # 白话 App
  ├── bili-infrastructure/ # Bili 基础设施
  ├── baihua-librechat/  # LibreChat 集成
  └── sing-app/          # Sing App

~/05-Entities/           # 实体
~/06-Infra/              # 基础设施
~/07-Archive/            # 归档
```

---

## 八、其他重要目录

| 目录 | 大小 | 说明 |
|------|------|------|
| `~/CAGI/` | — | CAGI 项目 |
| `~/CCAGI/` | — | CCAGI 项目 |
| `~/ENV/` | — | 环境配置 |
| `~/PROJECTS/` | — | 其他项目 |
| `~/archive/` | — | 归档 |
| `~/browser_use_env/` | — | Browser Use 环境 |
| `~/audio_separator_models/` | — | 音频分离模型 |
| `~/.ollama/` | — | Ollama 配置/模型 |
| `~/.cache/huggingface/` | — | HuggingFace 缓存 |
| `~/.cache/pip/` | — | pip 缓存 |
| `~/.local/lib/python3.10/site-packages/` | — | Python 包 |

---

## 九、Git 仓库

| 仓库 | 路径 | 说明 |
|------|------|------|
| `qwen3-loop-pruning` | `~/qwen3-loop-pruning/` | 本次实验脚本仓库 |
| `01-Principia/BDI` | `~/01-Principia/BDI/` | BDI 架构 |
| `01-Principia/The-Law-of-Circles` | `~/01-Principia/The-Law-of-Circles/` | 圆律理论 |
| `02-Brain/CrackNeural` | `~/02-Brain/CrackNeural/` | 神经网络破解 |
| `02-Brain/baihua` | `~/02-Brain/baihua/` | 白话 |
| `02-Brain/brain_v9` | `~/02-Brain/brain_v9/` | 大脑 v9 |
| `02-Brain/Brain` | `~/02-Brain/Brain/` | 大脑核心 |
| `03-Agents/BSEM` | `~/03-Agents/BSEM/` | BSEM |
| `03-Agents/CCAGI` | `~/03-Agents/CCAGI/` | CCAGI |
| `04-Products/baihua-app` | `~/04-Products/baihua-app/` | 白话 App |
| `04-Products/bili-infrastructure` | `~/04-Products/bili-infrastructure/` | Bili |
| `04-Products/baihua-librechat` | `~/04-Products/baihua-librechat/` | LibreChat |
| `04-Products/sing-app` | `~/04-Products/sing-app/` | Sing |

---

## 十、环境变量

```bash
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/usr/local/cuda
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=.../usr/local/cuda/lib64...
PATH=.../usr/local/cuda/bin:...miniconda3/bin:...local/bin...
HOME=/home/ubuntu
USER=ubuntu
SHELL=/usr/bin/zsh
TERM=xterm-256color
TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1
TRANSFORMERS_CACHE=~/.cache/huggingface
HF_HOME=~/.cache/huggingface
```

---

## 十一、网络与服务

| 服务 | 状态 | 端口 |
|------|------|------|
| frpc | 运行中（WSL→公网穿透） | 26026 |
| SSH | 通过 frpc 暴露 | 26026 |
| Ollama | 安装 | 默认 |
| Jupyter | 未运行 | — |
| TensorBoard | 未运行 | — |

---

## 十二、本次实验关键文件速查

### 核心脚本（GitHub 已同步）
```
~/modeling_qwen3_loop.py          # LoopAttention 模块
~/train_loop_manual.py            # 推荐训练脚本
~/train_with_loop.py              # Trainer API 版
~/analyze_activation.py           # 100 问题 FFN 分析
~/analyze_quick.py                # 20 问题快速分析
~/prune_ffn.py                    # 结构化剪枝 50%
~/test_pruned.py                  # 对比测试
~/generate_antiloop_data.py       # 反例数据构造
~/preprocess_loop_data.py         # 数据预处理
~/diag_inference.py               # 推理诊断
```

### 基座模型
```
~/dynamic_infinite_train_v2/final/     # 1.2G, Qwen3-0.6B SFT
```

### 训练数据
```
/tmp/training-data/sft_final_v3_p1.jsonl    # 3000 条
/tmp/training-data/sft_final_v3_p2.jsonl    # 3000 条
/tmp/training-data/sft_antiloop.jsonl       # 5415 条
```

### 日志
```
~/dynamic_infinite_train_v2_nohup.log   # 基座训练日志
~/analyze_quick_nohup.log               # 激活分析日志（运行中）
~/loop_manual_nohup.log                   # 手动训练日志
```

---

## 十三、已知问题与限制

1. **WSL 内存峰值**: `dataset.map()` 在 6884 条样本时 OOM，需用手写 DataLoader
2. **frpc 不稳定**: 长连接会随机断开，所有操作必须后台 + 日志
3. **Python stdout 缓冲**: 默认全缓冲，需 `PYTHONUNBUFFERED=1`
4. **显存无隔离**: WSL 下 Ollama 直接操作 Windows GPU，大模型会蓝屏
5. **GitHub 推送超时**: frpc 网络不稳定，通过 API 上传文件

---

*本盘点由人机协同实验组生成，用于 WSL → AutoDL 迁移参考。*
