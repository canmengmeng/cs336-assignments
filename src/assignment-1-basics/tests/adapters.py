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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return an AdamW optimizer class (not a torch.optim.AdamW instance).

    Returns:
        A class that can be instantiated with model parameters and hyperparameters
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
