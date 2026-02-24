# DITSB-v2 大语言模型 (LLM) 训练方案

本目录包含了基于 DITSB-v2 架构 (Grand Unified Manifold Framework) 训练大型语言模型所需的完整方案、配置与脚本。这里加入了对连续流文本生成的详尽优化与日志追踪。

## 方案目录与适用范围

- `README.md`: 训练方案的整体使用说明。
- `config_7b.yaml`: 包含模型尺寸 (7B)、硬件规格预测参数、多节点日志输出以及 Sinkhorn/CTMC 核心设定的配置文件。
- `prepare_data.py`: 基于 HuggingFace `datasets` 与 `tokenizers` 的高效分布式数据预处理脚手架。
- `predict_performance.py`: **新增** 能够以 Chinchilla Scaling Laws 结合连续流 MSE 收敛来量化硬件时间、预测 FLOPs 利用率及最终 PPL 表现的评估工具。
- `train_llm.py`: **已增强** 包含内置动态吞吐量 (Tok/s)、剩余时间ETA计算和全套基于 `logging` 输出的大模型分布式训练挂载脚本。

## 性能与用时预测 (Performance Prediction)

在开启数百张显卡进行漫长的预训练之前，可预先测算理论时间及模型表现：
```bash
python llm_training/predict_performance.py --config llm_training/config_7b.yaml
```
- 该脚本会读取诸如硬件 H100 架构类型、预设架构（参数量估算约6.9B）、总 token 数。
- 精准预测出基于给定 MFU (Model FLOPs Utilization) 期望的完成 **天数** 以及收敛后的连续 CTMC 验证集 Loss。

## 1. 核心理论架构适配

与传统的左到右自回归预训练模型（如 LLaMA, GPT）不同，DITSB-LLM 是一次纯通过 ODE/SDE 集成的前向平流模型：
- 摒弃 Casual Attention Mask，代之以全双工 Bidirectional Attention。
- 梯度计算不依赖传统反向传播，完全依赖数学伴随（Adjoint）反向积分实现 $O(1)$ 显存占用开销，允许配置中直接解禁两万以上的高强度上下文 `max_seq_len`。

## 2. 训练数据准备

使用脚本准备数据：
```bash
python llm_training/prepare_data.py --dataset openwebtext --tokenizer_name unsloth/llama-3-8b --seq_len 2048 --output_dir ./data/openwebtext_processed
```

## 3. 分布式训练与日志监控 (`train_llm.py`)

实际执行模型的训练过程（脚本已加装丰富的 Logging 输出）：
```bash
accelerate launch llm_training/train_llm.py llm_training/config_7b.yaml
```

**训练日志系统亮点**:
- 动态输出 `Tokens / Sec` 了解硬件瓶颈。
- 依据 `config_7b.yaml` 中规定的 `max_steps` 推算出精确到秒的 `ETA` (Estimated Time of Arrival)。
- `PPL(approx)`：动态将分类概率流向量场中的目标导数映射至直观的句子困惑度，用作平滑监测。
- 自动利用 `clip_grad_norm` 防治离散空间连续映射早期的梯度爆炸问题。
