# CS336 Assignment 1: Basics — Language Modeling From Scratch

## 核心概念

本作业从零构建一个完整的 Transformer 语言模型，包含三大核心组件：

### 1. BPE Tokenizer
- **Byte Pair Encoding**: 从字节级别构建词表，通过迭代合并最高频的字节对
- **核心算法**: 统计 → 合并 → 重复，直到达到目标词表大小
- **公式**: `vocab_size = 256 (base bytes) + num_merges + len(special_tokens)`

### 2. Transformer 架构
- **Pre-norm Transformer**: 使用 RMSNorm 进行层归一化
- **RoPE**: 旋转位置编码，将位置信息编码到 Q/K 向量中
- **SwiGLU**: 改进的 FFN 激活函数，比 ReLU 效果更好
- **权重绑定**: LM head 与 embedding 层共享权重

### 3. 训练基础设施
- **AdamW**: 解耦权重衰减的优化器
- **Cosine LR Schedule**: 带线性预热的余弦退火学习率
- **Gradient Clipping**: 防止梯度爆炸

## 作业结构

```
src/assignment-1-basics/
├── cs336_basics/              # 核心实现包
│   ├── __init__.py
│   └── pretokenization_example.py
├── tests/                     # 测试文件
│   └── adapters.py           # 学生实现的接口文件
├── data/                      # 数据集目录
├── pyproject.toml             # Python 项目配置
└── README.md                  # 本文件
```

## 实现任务清单

### Part 1: 核心组件 (5 个)
- [ ] `run_linear` — 线性变换 y = x @ W^T
- [ ] `run_embedding` — 嵌入查找
- [ ] `run_rmsnorm` — RMSNorm 归一化
- [ ] `run_silu` — SiLU 激活函数
- [ ] `run_softmax` — Softmax 函数

### Part 2: 注意力机制 (4 个)
- [ ] `run_scaled_dot_product_attention` — 缩放点积注意力
- [ ] `run_multihead_self_attention` — 多头自注意力
- [ ] `run_rope` — 旋转位置编码
- [ ] `run_multihead_self_attention_with_rope` — 带 RoPE 的多头自注意力

### Part 3: Transformer 架构 (2 个)
- [ ] `run_transformer_block` — 单个 Transformer 块
- [ ] `run_transformer_lm` — 完整 Transformer 语言模型

### Part 4: 训练组件 (5 个)
- [ ] `run_cross_entropy` — 交叉熵损失
- [ ] `run_get_batch` — 批数据采样
- [ ] `run_gradient_clipping` — 梯度裁剪
- [ ] `get_adamw_cls` — AdamW 优化器
- [ ] `run_get_lr_cosine_schedule` — 余弦学习率调度

### Part 5: 检查点管理 (2 个)
- [ ] `run_save_checkpoint` — 保存检查点
- [ ] `run_load_checkpoint` — 加载检查点

### Part 6: 分词器 (2 个)
- [ ] `get_tokenizer` — BPE 分词器构建
- [ ] `run_train_bpe` — BPE 训练

## 运行命令

### 环境设置
```bash
# 安装依赖
cd src/assignment-1-basics
uv sync

# 下载数据集 (TinyStories / OpenWebText)
# 参考课程说明下载数据到 data/ 目录
```

### 运行测试
```bash
cd src/assignment-1-basics
uv run pytest
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer LM                           │
├─────────────────────────────────────────────────────────────┤
│  Input: token_ids (batch, seq_len)                          │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Token Embedding  │ (vocab_size × d_model)                │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Transformer Block × N                   │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │  RMSNorm → MultiHead Attention + Residual   │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │  RMSNorm → SwiGLU FFN + Residual            │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    RMSNorm       │                                       │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   LM Head        │ (权重绑定: embed.T)                   │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  Output: logits (batch, seq_len, vocab_size)                │
└─────────────────────────────────────────────────────────────┘
```

## 参考资源

- [课程网站](https://cs336.stanford.edu)
- [官方作业仓库](https://github.com/stanford-cs336/assignment1-basics)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202)
