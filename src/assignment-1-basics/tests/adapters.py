"""CS336 Assignment 1: Test Adapters

This file contains stub implementations for all the components needed to build
a Transformer language model from scratch. Each function raises NotImplementedError
and must be implemented by the student.

Functions are organized into:
1. Core Neural Network Components (Linear, Embedding, SwiGLU, Attention)
2. Transformer Architecture (Block, Full LM)
3. Utility Functions (RMSNorm, SiLU, Softmax, CrossEntropy)
4. Training Infrastructure (Gradient Clipping, AdamW, LR Schedule, Checkpointing)
5. Tokenization (BPE Tokenizer, BPE Training)
"""

from __future__ import annotations

import os
from typing import IO, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. CORE NEURAL NETWORK COMPONENTS
# ============================================================================

def run_linear(
    d_in: int,
    d_out: int,
    weights: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Compute a linear transformation: y = x @ W^T

    Args:
        d_in: Input dimension
        d_out: Output dimension
        weights: Weight matrix of shape (d_out, d_in)
        in_features: Input tensor of shape (*, d_in)

    Returns:
        Output tensor of shape (*, d_out)
    """
    # Einstein 求和: j 收缩，i 保留，... 保持 batch 维度
    # (*, d_in) @ (d_out, d_in) -> (*, d_out)
    return torch.einsum('...j,ij->...i', in_features, weights)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Look up embeddings for given token IDs.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension
        weights: Embedding weight matrix of shape (vocab_size, d_model)
        token_ids: Token IDs to look up, shape (*,)

    Returns:
        Embedded vectors of shape (*, d_model)
    """
    # 花式索引: 用 token_ids 作为行索引，取出对应向量
    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Implement the SwiGLU feed-forward network.

    SwiGLU(x) = (SiLU(x @ W1^T) ⊙ (x @ W3^T)) @ W2^T

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        w1_weight: Weight matrix for gate projection, shape (d_ff, d_model)
        w2_weight: Weight matrix for down projection, shape (d_model, d_ff)
        w3_weight: Weight matrix for up projection, shape (d_ff, d_model)
        in_features: Input tensor of shape (*, d_model)

    Returns:
        Output tensor of shape (*, d_model)
    """
    # 1. Gate 路径: SiLU(x @ W1^T)  (*, d_model) → (*, d_ff)
    gate = run_silu(torch.einsum('...j,ij->...i', in_features, w1_weight))
    # 2. Up 路径: x @ W3^T          (*, d_model) → (*, d_ff)
    up = torch.einsum('...j,ij->...i', in_features, w3_weight)
    # 3. 门控: gate ⊙ up            (*, d_ff) ⊙ (*, d_ff) → (*, d_ff)
    gated = torch.einsum('...i,...i->...i', gate, up)
    # 4. Down 投影: gated @ W2^T    (*, d_ff) → (*, d_model)
    return torch.einsum('...j,ij->...i', gated, w2_weight)


def run_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, d_k)
        k: Key tensor of shape (batch, num_heads, seq_len, d_k)
        v: Value tensor of shape (batch, num_heads, seq_len, d_v)
        mask: Optional boolean mask of shape (batch, 1, seq_len, seq_len)
              where True indicates positions to attend to

    Returns:
        Attention output of shape (batch, num_heads, seq_len, d_v)
    """
    d_k = q.shape[-1]
    # 1. Q @ K^T / √d_k: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
    #    einsum: ...ik,...jk->...ij  (i=query位置, j=key位置, k=特征维度)
    scores = torch.einsum('...ik,...jk->...ij', q, k) / (d_k ** 0.5)
    # 2. Mask: False 的位置设为 -∞，softmax 后变成 0
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    # 3. Softmax: 沿 key 维度归一化
    weights = run_softmax(scores, dim=-1)
    # 4. 加权求和: weights @ V
    #    einsum: ...ij,...jk->...ik  (i=query位置, j=key位置, k=特征维度)
    return torch.einsum('...ij,...jk->...ik', weights, v)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Multi-head self-attention without RoPE.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        q_proj_weight: Query projection weight, shape (d_model, d_model)
        k_proj_weight: Key projection weight, shape (d_model, d_model)
        v_proj_weight: Value projection weight, shape (d_model, d_model)
        o_proj_weight: Output projection weight, shape (d_model, d_model)
        in_features: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads

    # 1. 线性投影: (*, d_model) → (*, d_model)
    Q = torch.einsum('...i,ji->...j', in_features, q_proj_weight)
    K = torch.einsum('...i,ji->...j', in_features, k_proj_weight)
    V = torch.einsum('...i,ji->...j', in_features, v_proj_weight)

    # 2. 拆分多头: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    # 3. 缩放点积注意力
    attn_output = run_scaled_dot_product_attention(Q, K, V)

    # 4. 合并多头: (batch, num_heads, seq, d_k) → (batch, seq, d_model)
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)

    # 5. 输出投影
    return torch.einsum('...i,ji->...j', attn_output, o_proj_weight)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-head self-attention with Rotary Position Embeddings (RoPE).

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for RoPE precomputation
        theta: Base frequency for RoPE
        q_proj_weight: Query projection weight, shape (d_model, d_model)
        k_proj_weight: Key projection weight, shape (d_model, d_model)
        v_proj_weight: Value projection weight, shape (d_model, d_model)
        o_proj_weight: Output projection weight, shape (d_model, d_model)
        in_features: Input tensor of shape (batch, seq_len, d_model)
        token_positions: Optional position indices, shape (batch, seq_len)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads

    # 1. 线性投影
    Q = torch.einsum('...i,ji->...j', in_features, q_proj_weight)
    K = torch.einsum('...i,ji->...j', in_features, k_proj_weight)
    V = torch.einsum('...i,ji->...j', in_features, v_proj_weight)

    # 2. 拆分多头: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    # 3. 对 Q 和 K 应用 RoPE（每个头独立处理）
    #    RoPE 输入形状: (batch, seq, d_k)
    #    需要 reshape: (batch * num_heads, seq, d_k)
    Q_flat = Q.reshape(batch * num_heads, seq_len, d_k)
    K_flat = K.reshape(batch * num_heads, seq_len, d_k)

    Q_rotated = run_rope(d_k, theta, max_seq_len, Q_flat, token_positions)
    K_rotated = run_rope(d_k, theta, max_seq_len, K_flat, token_positions)

    # reshape 回多头形状
    Q = Q_rotated.reshape(batch, num_heads, seq_len, d_k)
    K = K_rotated.reshape(batch, num_heads, seq_len, d_k)

    # 4. 缩放点积注意力
    attn_output = run_scaled_dot_product_attention(Q, K, V)

    # 5. 合并多头
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)

    # 6. 输出投影
    return torch.einsum('...i,ji->...j', attn_output, o_proj_weight)


def run_rope(
    d_model: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to a query or key tensor.

    Args:
        d_model: Model dimension (must equal head_dim for proper RoPE)
        theta: Base frequency
        max_seq_len: Maximum sequence length
        in_query_or_key: Input tensor of shape (batch, seq_len, d_model)
        token_positions: Optional position indices, shape (batch, seq_len)

    Returns:
        Rotated tensor of same shape as input
    """
    # 1. 计算频率: θ_i = theta^(-2i/d_model), i = 0, 1, ..., d_model/2 - 1
    #    freqs 形状: (d_model/2,)
    i = torch.arange(d_model // 2, device=in_query_or_key.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2 * i / d_model))
    # 2. 计算旋转角度: positions × freqs
    #    positions: (batch, seq_len) → (batch, seq_len, 1)
    #    freqs: (d_model/2,) → (1, 1, d_model/2)
    #    angles: (batch, seq_len, d_model/2)
    if token_positions is None:
        token_positions = torch.arange(in_query_or_key.shape[1], device=in_query_or_key.device)
    angles = token_positions.float()[..., None] * freqs
    # 3. 构造旋转: cos 和 sin
    cos_angles = torch.cos(angles)  # (batch, seq_len, d_model/2)
    sin_angles = torch.sin(angles)  # (batch, seq_len, d_model/2)
    # 4. 拆分输入为偶数和奇数维度
    #    x_even: [..., 0, 2, 4, ...],  x_odd: [..., 1, 3, 5, ...]
    x_even = in_query_or_key[..., 0::2]  # (batch, seq_len, d_model/2)
    x_odd = in_query_or_key[..., 1::2]   # (batch, seq_len, d_model/2)
    # 5. 应用旋转矩阵:
    #    [x_even'] = cos * x_even - sin * x_odd
    #    [x_odd']  = sin * x_even + cos * x_odd
    out_even = cos_angles * x_even - sin_angles * x_odd
    out_odd = sin_angles * x_even + cos_angles * x_odd
    # 6. 交错合并回原形状
    result = torch.stack([out_even, out_odd], dim=-1)  # (batch, seq, d/2, 2)
    result = result.reshape(in_query_or_key.shape)
    return result


# ============================================================================
# 2. TRANSFORMER ARCHITECTURE
# ============================================================================

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Run a pre-norm Transformer block with RoPE.

    Architecture:
        x = in_features
        x = x + MultiHeadAttention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        weights: Dictionary containing:
            - attn.q_proj.weight: (d_model, d_model)
            - attn.k_proj.weight: (d_model, d_model)
            - attn.v_proj.weight: (d_model, d_model)
            - attn.o_proj.weight: (d_model, d_model)
            - ln1.weight: (d_model,)
            - ffn.w1.weight: (d_ff, d_model)
            - ffn.w2.weight: (d_model, d_ff)
            - ffn.w3.weight: (d_ff, d_model)
            - ln2.weight: (d_model,)
        in_features: Input tensor of shape (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    # 1. 注意力子层: x + MHA(RMSNorm(x))
    normed1 = run_rmsnorm(d_model, 1e-5, weights['ln1.weight'], in_features)
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.o_proj.weight'],
        normed1
    )
    h = in_features + attn_out

    # 2. FFN 子层: h + SwiGLU(RMSNorm(h))
    normed2 = run_rmsnorm(d_model, 1e-5, weights['ln2.weight'], h)
    ffn_out = run_swiglu(
        d_model, d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        normed2
    )
    return h + ffn_out


def run_transformer_lm(
    vocab_size: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,
) -> torch.Tensor:
    """Full forward pass of a Transformer language model.

    Architecture:
        x = TokenEmbedding(in_indices)
        for each layer:
            x = TransformerBlock(x)
        x = RMSNorm(x)
        logits = x @ embed_weights^T  (weight tying)

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        weights: Dictionary containing all model weights
        in_indices: Token indices of shape (batch, seq_len)

    Returns:
        Unnormalized logits of shape (batch, seq_len, vocab_size)
    """
    # 1. Token 嵌入: (batch, seq) → (batch, seq, d_model)
    x = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)

    # 2. N 层 Transformer Block
    for i in range(num_layers):
        layer_weights = {
            'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
            'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
            'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
            'attn.o_proj.weight': weights[f'layers.{i}.attn.o_proj.weight'],
            'ln1.weight': weights[f'layers.{i}.ln1.weight'],
            'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
            'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
            'ffn.w3.weight': weights[f'layers.{i}.ffn.w3.weight'],
            'ln2.weight': weights[f'layers.{i}.ln2.weight'],
        }
        x = run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, layer_weights, x)

    # 3. 最终 RMSNorm
    x = run_rmsnorm(d_model, 1e-5, weights['ln_final.weight'], x)

    # 4. LM Head (权重绑定): logits = x @ embed_weights^T
    logits = torch.einsum('...i,ji->...j', x, weights['token_embeddings.weight'])

    return logits


# ============================================================================
# 3. UTILITY FUNCTIONS
# ============================================================================

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    """Apply RMSNorm.

    RMSNorm(x) = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability
        weights: Learnable weights of shape (d_model,)
        in_features: Input tensor of shape (*, d_model)

    Returns:
        Normalized tensor of same shape
    """
    # 1. 计算每个向量的均方根 RMS(x) = sqrt(mean(x²) + ε)
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    # 2. 归一化并缩放: x / RMS(x) · γ
    return (in_features / rms) * weights


def run_silu(in_features: torch.Tensor) -> torch.Tensor:
    """Apply SiLU activation (x * sigmoid(x)).

    Args:
        in_features: Input tensor

    Returns:
        Activated tensor of same shape
    """
    return in_features * torch.sigmoid(in_features)


def run_softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply softmax along a specified dimension.

    Args:
        in_features: Input tensor
        dim: Dimension to apply softmax

    Returns:
        Softmax output of same shape
    """
    # 数值稳定: 减去最大值防止 exp 溢出
    max_vals = in_features.max(dim=dim, keepdim=True).values
    exp_vals = torch.exp(in_features - max_vals)
    return exp_vals / exp_vals.sum(dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute average cross-entropy loss over a batch.

    CE(inputs, targets) = -mean(log(softmax(inputs)[targets]))

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size)
        targets: Target class indices of shape (batch_size,)

    Returns:
        Scalar loss tensor
    """
    # 数值稳定的 log_softmax: x_i - max(x) - log(Σ exp(x_j - max(x)))
    max_vals = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(inputs - max_vals).sum(dim=-1)) + max_vals.squeeze(-1)
    # log_softmax = logits - log_sum_exp
    log_softmax = inputs - log_sum_exp.unsqueeze(-1)
    # 取正确位置的 log 概率，取负，求均值
    # gather 展开: log_softmax[i, targets[i]]
    batch_size = inputs.shape[0]
    loss = -log_softmax[torch.arange(batch_size), targets].mean()
    return loss


def run_get_batch(
    dataset: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input/label pairs from a 1D dataset.

    Args:
        dataset: 1D tensor of token IDs
        batch_size: Number of samples per batch
        context_length: Length of each sequence
        device: Device to place tensors on

    Returns:
        Tuple of (inputs, labels) each of shape (batch_size, context_length)
    """
    # 1. 随机选起始位置: 最大起始点 = N - context_length - 1
    max_start = len(dataset) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    # 2. 构造索引矩阵: 每行是 [start, start+1, ..., start+ctx-1]
    offsets = torch.arange(context_length)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (batch, ctx)
    # 3. 输入和标签错位一位
    inputs = dataset[indices].to(device)
    labels = dataset[indices + 1].to(device)
    return inputs, labels


# ============================================================================
# 4. TRAINING INFRASTRUCTURE
# ============================================================================

def run_gradient_clipping(
    parameters: torch.nn.Parameter | Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip combined gradient L2 norm of parameters (in-place).

    If total_norm > max_l2_norm, scale all gradients by max_l2_norm / total_norm.

    Args:
        parameters: Single parameter or iterable of parameters
        max_l2_norm: Maximum allowed L2 norm
    """
    # 1. 统一处理：单个参数也变成列表
    if isinstance(parameters, torch.nn.Parameter):
        params = [parameters]
    else:
        params = list(parameters)

    # 2. 收集有梯度的参数
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    # 3. 计算总 L2 范数: √(Σ ‖gᵢ‖²)
    total_norm_sq = sum(torch.sum(g ** 2) for g in grads)
    total_norm = torch.sqrt(total_norm_sq)

    # 4. 如果超过阈值，等比例缩放
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)  # 原地缩放


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return an AdamW optimizer class (not a torch.optim.AdamW instance).

    Returns:
        A class that can be instantiated with model parameters and hyperparameters
    """
    class AdamW(torch.optim.Optimizer):
        """AdamW optimizer with decoupled weight decay."""

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    # 初始化状态
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)  # 一阶动量
                        state['v'] = torch.zeros_like(p)  # 二阶动量

                    m, v = state['m'], state['v']
                    state['step'] += 1
                    t = state['step']

                    # 更新动量
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                    # 偏差修正
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    # AdamW: 解耦权重衰减
                    # 1. 先衰减参数: θ = θ · (1 - lr · λ)
                    p.mul_(1 - lr * weight_decay)
                    # 2. 再用 Adam 更新: θ = θ - lr · m̂/(√v̂ + ε)
                    p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        it: Current iteration number
        max_learning_rate: Peak learning rate after warmup
        min_learning_rate: Minimum learning rate
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Number of cosine annealing iterations

    Returns:
        Learning rate for the current iteration
    """
    import math

    # 阶段 1: 线性预热
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # 阶段 2: 余弦退火
    if it <= cosine_cycle_iters:
        t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * t))

    # 阶段 3: 保持最小学习率
    return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | IO[bytes],
) -> None:
    """Save model, optimizer, and iteration state.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        iteration: Current training iteration
        out: File path or file-like object to save to
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    f: str | os.PathLike | IO[bytes],
) -> int:
    """Load model, optimizer, and iteration state.

    Args:
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into
        f: File path or file-like object to load from

    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(f, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']


# ============================================================================
# 5. TOKENIZATION
# ============================================================================

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Construct a BPE tokenizer from vocab, merges, and optional special tokens.

    Args:
        vocab: Dictionary mapping token IDs to byte sequences
        merges: List of merge operations as (bytes, bytes) pairs
        special_tokens: Optional list of special token strings

    Returns:
        A tokenizer instance
    """
    class Tokenizer:
        def __init__(self, vocab, merges, special_tokens):
            self.vocab = vocab                          # id → bytes
            self.vocab_reverse = {v: k for k, v in vocab.items()}  # bytes → id
            self.merges = merges
            self.special_tokens = special_tokens or []
            # 合并优先级: merges 列表中越靠前优先级越高
            self.merge_priority = {(a, b): i for i, (a, b) in enumerate(merges)}

        def encode(self, text: str) -> list[int]:
            """编码: 文本 → token IDs"""
            # 1. 处理特殊 token
            tokens = []
            # 先按特殊 token 分割
            parts = [text]
            for st in self.special_tokens:
                new_parts = []
                for part in parts:
                    if part == st:
                        new_parts.append(st)
                    else:
                        new_parts.extend(part.split(st))
                parts = new_parts

            # 2. 对每个非特殊部分应用 BPE
            for part in parts:
                if part in self.special_tokens:
                    tokens.append(self.vocab_reverse[part.encode('utf-8')])
                else:
                    tokens.extend(self._bpe_encode(part))

            return tokens

        def _bpe_encode(self, text: str) -> list[int]:
            """对普通文本应用 BPE 编码"""
            # 转成字节序列
            symbols = [bytes([b]) for b in text.encode('utf-8')]

            # 反复合并直到无法合并
            while len(symbols) > 1:
                # 找最高优先级的可合并 pair
                best_pair = None
                best_priority = float('inf')

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if pair in self.merge_priority:
                        priority = self.merge_priority[pair]
                        if priority < best_priority:
                            best_priority = priority
                            best_pair = pair

                if best_pair is None:
                    break  # 没有可合并的 pair

                # 执行合并
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            # 查词表得到 token IDs
            return [self.vocab_reverse[s] for s in symbols]

        def decode(self, ids: list[int]) -> str:
            """解码: token IDs → 文本"""
            result = b''
            for token_id in ids:
                result += self.vocab[token_id]
            return result.decode('utf-8', errors='replace')

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer from an input corpus.

    Args:
        input_path: Path to training corpus file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include

    Returns:
        Tuple of (vocab, merges):
            - vocab: Dictionary mapping token IDs to byte sequences
            - merges: List of merge operations as (bytes, bytes) pairs
    """
    from collections import Counter

    # 1. 读取语料
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 初始化词表: 256 个字节 + 特殊 token
    vocab = {}
    token_id = 0

    # 添加特殊 token
    for st in special_tokens:
        vocab[token_id] = st.encode('utf-8')
        token_id += 1

    # 添加 256 个基础字节
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    # 3. 把语料拆成字节序列（按行处理，每行独立）
    lines = text.split('\n')
    corpus = []
    for line in lines:
        line_bytes = line.encode('utf-8')
        # 每行拆成字节序列
        word = tuple(bytes([b]) for b in line_bytes)
        corpus.append(word)

    # 4. 循环合并直到达到词表大小
    merges = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        # a. 统计所有相邻 pair 频率
        pair_counts = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        if not pair_counts:
            break  # 没有可合并的 pair

        # b. 找最高频 pair
        best_pair = max(pair_counts, key=pair_counts.get)

        # c. 合并: 更新语料表示
        new_token = best_pair[0] + best_pair[1]
        new_corpus = []
        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(tuple(new_word))
        corpus = new_corpus

        # d. 添加到词表和合并规则
        vocab[token_id] = new_token
        merges.append(best_pair)
        token_id += 1

    return vocab, merges
