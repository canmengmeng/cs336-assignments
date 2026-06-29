# CS336 Stanford 课程作业

斯坦福 CS336: Language Modeling from Scratch 课程作业实现。

## 作业概览

| 作业 | 主题 | 状态 |
|------|------|------|
| Assignment 1 | Basics — 从零构建 Transformer LM | 🚧 进行中 |
| Assignment 2 | Systems — 性能优化与分布式训练 | ⏳ 待开始 |
| Assignment 3 | Scaling — 缩放定律与训练 | ⏳ 待开始 |
| Assignment 4 | Data — 数据处理与去重 | ⏳ 待开始 |
| Assignment 5 | Alignment — SFT 与 RLHF | ⏳ 待开始 |

## 项目结构

```
cs336-assignments/
├── src/
│   ├── assignment-1-basics/     # Assignment 1: 基础组件实现
│   │   ├── cs336_basics/        # 核心 Python 包
│   │   ├── tests/               # 测试文件
│   │   ├── data/                # 数据集目录
│   │   ├── README.md            # 作业说明
│   │   └── pyproject.toml       # Python 配置
│   ├── assignment-2-systems/    # Assignment 2: 系统优化
│   ├── assignment-3-scaling/    # Assignment 3: 缩放定律
│   ├── assignment-4-data/       # Assignment 4: 数据处理
│   └── assignment-5-alignment/  # Assignment 5: 对齐训练
├── CLAUDE.md                    # AI 协作指南
└── README.md                    # 本文件
```

## 环境配置

本项目使用 [uv](https://github.com/astral-sh/uv) 进行 Python 环境管理：

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 进入 Assignment 1 目录
cd src/assignment-1-basics

# 安装依赖
uv sync

# 运行测试
uv run pytest
```

## Assignment 1: Basics

从零实现完整的 Transformer 语言模型，包括：

1. **BPE Tokenizer** — 字节对编码分词器
2. **Transformer 架构** — 包含 RoPE、SwiGLU、RMSNorm
3. **训练基础设施** — AdamW、余弦学习率、梯度裁剪
4. **检查点管理** — 模型保存与加载

### 实现进度 (8/20)

| Part | 任务 | 状态 |
|------|------|------|
| Part 1 | 核心组件 (5) | ✅ 全部完成 |
| Part 2 | 注意力机制 (4) | ⏳ 待开始 |
| Part 3 | Transformer 架构 (2) | ⏳ 待开始 |
| Part 4 | 训练组件 (5) | 🚧 进行中 (2/5) |
| Part 5 | 检查点管理 (2) | ⏳ 待开始 |
| Part 6 | 分词器 (2) | ⏳ 待开始 |

**已完成的核心组件：**
- ✅ `run_silu` — SiLU 激活函数
- ✅ `run_softmax` — Softmax 函数
- ✅ `run_rmsnorm` — RMSNorm 归一化
- ✅ `run_linear` — 线性变换
- ✅ `run_embedding` — 嵌入查找
- ✅ `run_cross_entropy` — 交叉熵损失
- ✅ `run_get_batch` — 批数据采样
- ✅ `run_swiglu` — SwiGLU 激活函数

详细说明见 [assignment-1-basics/README.md](src/assignment-1-basics/README.md)

## 学习资源

- [CS336 课程网站](https://cs336.stanford.edu)
- [Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
