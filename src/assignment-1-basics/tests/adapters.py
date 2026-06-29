from __future__ import annotations

import os
import math
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """Compute linear transformation: y = x @ W^T"""
    return torch.einsum('...j,ij->...i', in_features, weights)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """Look up embeddings for token IDs"""
    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """SwiGLU: (SiLU(x @ W1^T) ⊙ (x @ W3^T)) @ W2^T"""
    gate = run_silu(torch.einsum('...j,ij->...i', in_features, w1_weight))
    up = torch.einsum('...j,ij->...i', in_features, w3_weight)
    gated = torch.einsum('...i,...i->...i', gate, up)
    return torch.einsum('...j,ij->...i', gated, w2_weight)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention"""
    d_k = Q.shape[-1]
    scores = torch.einsum('...ik,...jk->...ij', Q, K) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = run_softmax(scores, dim=-1)
    return torch.einsum('...ij,...jk->...ik', weights, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence d_model"],
) -> Float[Tensor, " ... sequence d_model"]:
    """Multi-head self-attention without RoPE"""
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads

    Q = torch.einsum('...i,ji->...j', in_features, q_proj_weight)
    K = torch.einsum('...i,ji->...j', in_features, k_proj_weight)
    V = torch.einsum('...i,ji->...j', in_features, v_proj_weight)

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    attn_output = run_scaled_dot_product_attention(Q, K, V)

    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    return torch.einsum('...i,ji->...j', attn_output, o_proj_weight)


def run_rope(
    d_model: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence d_model"],
    token_positions: Int[Tensor, " ... sequence"] | None = None,
) -> Float[Tensor, " ... sequence d_model"]:
    """Apply Rotary Position Embeddings (RoPE)"""
    i = torch.arange(d_model // 2, device=in_query_or_key.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2 * i / d_model))

    if token_positions is None:
        token_positions = torch.arange(in_query_or_key.shape[-2], device=in_query_or_key.device)

    angles = token_positions.float()[..., None] * freqs
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]

    out_even = cos_angles * x_even - sin_angles * x_odd
    out_odd = sin_angles * x_even + cos_angles * x_odd

    result = torch.stack([out_even, out_odd], dim=-1)
    return result.reshape(in_query_or_key.shape)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence d_model"],
    token_positions: Int[Tensor, " ... sequence"] | None = None,
) -> Float[Tensor, " ... sequence d_model"]:
    """Multi-head self-attention with RoPE"""
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads

    Q = torch.einsum('...i,ji->...j', in_features, q_proj_weight)
    K = torch.einsum('...i,ji->...j', in_features, k_proj_weight)
    V = torch.einsum('...i,ji->...j', in_features, v_proj_weight)

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    Q_flat = Q.reshape(batch * num_heads, seq_len, d_k)
    K_flat = K.reshape(batch * num_heads, seq_len, d_k)

    # Expand token_positions for each head: (*, seq) -> (batch*heads, seq)
    if token_positions is not None:
        # Broadcast to (batch, seq) first, then expand for heads
        pos = token_positions.expand(batch, seq_len)  # (batch, seq)
        rope_positions = pos.unsqueeze(1).expand(-1, num_heads, -1).reshape(batch * num_heads, seq_len)
    else:
        rope_positions = None

    Q_rotated = run_rope(d_k, theta, max_seq_len, Q_flat, rope_positions)
    K_rotated = run_rope(d_k, theta, max_seq_len, K_flat, rope_positions)

    Q = Q_rotated.reshape(batch, num_heads, seq_len, d_k)
    K = K_rotated.reshape(batch, num_heads, seq_len, d_k)

    attn_output = run_scaled_dot_product_attention(Q, K, V)

    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    return torch.einsum('...i,ji->...j', attn_output, o_proj_weight)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " ... sequence d_model"],
) -> Float[Tensor, " ... sequence d_model"]:
    """Pre-norm Transformer block with RoPE"""
    normed1 = run_rmsnorm(d_model, 1e-5, weights['ln1.weight'], in_features)
    # Support both 'o_proj' and 'output_proj' key names
    o_proj_key = 'attn.output_proj.weight' if 'attn.output_proj.weight' in weights else 'attn.o_proj.weight'
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights[o_proj_key],
        normed1
    )
    h = in_features + attn_out

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
    context_length: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " ... sequence"],
) -> Float[Tensor, " ... sequence vocab_size"]:
    """Full Transformer language model forward pass"""
    x = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)

    for i in range(num_layers):
        # Support both 'o_proj' and 'output_proj' key names
        o_proj_key = f'layers.{i}.attn.output_proj.weight' if f'layers.{i}.attn.output_proj.weight' in weights else f'layers.{i}.attn.o_proj.weight'
        layer_weights = {
            'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
            'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
            'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
            'attn.output_proj.weight': weights[o_proj_key],
            'ln1.weight': weights[f'layers.{i}.ln1.weight'],
            'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
            'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
            'ffn.w3.weight': weights[f'layers.{i}.ffn.w3.weight'],
            'ln2.weight': weights[f'layers.{i}.ln2.weight'],
        }
        x = run_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, x)

    x = run_rmsnorm(d_model, 1e-5, weights['ln_final.weight'], x)
    logits = torch.einsum('...i,ji->...j', x, weights['token_embeddings.weight'])
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """RMSNorm: x / RMS(x) * weight"""
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    return (in_features / rms) * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """SiLU: x * sigmoid(x)"""
    return in_features * torch.sigmoid(in_features)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """Numerically stable softmax"""
    max_vals = in_features.max(dim=dim, keepdim=True).values
    exp_vals = torch.exp(in_features - max_vals)
    return exp_vals / exp_vals.sum(dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Cross-entropy loss"""
    max_vals = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(inputs - max_vals).sum(dim=-1)) + max_vals.squeeze(-1)
    log_softmax = inputs - log_sum_exp.unsqueeze(-1)
    batch_size = inputs.shape[0]
    return -log_softmax[torch.arange(batch_size), targets].mean()


def run_get_batch(
    dataset: Int[Tensor, " dataset_size"],
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[Int[Tensor, " batch_size context_length"], Int[Tensor, " batch_size context_length"]]:
    """Sample random input/label pairs from a 1D dataset"""
    if not isinstance(dataset, torch.Tensor):
        dataset = torch.tensor(dataset)
    max_start = len(dataset) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    offsets = torch.arange(context_length)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    inputs = dataset[indices].to(device)
    labels = dataset[indices + 1].to(device)
    return inputs, labels


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradient L2 norm (in-place)"""
    if isinstance(parameters, torch.nn.Parameter):
        params = [parameters]
    else:
        params = list(parameters)

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    total_norm_sq = sum(torch.sum(g ** 2) for g in grads)
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """Return AdamW optimizer class"""

    class AdamW(torch.optim.Optimizer):
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

                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)

                    m, v = state['m'], state['v']
                    state['step'] += 1
                    t = state['step']

                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    p.mul_(1 - lr * weight_decay)
                    p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup"""
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it <= cosine_cycle_iters:
        t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * t))

    return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | IO[bytes],
) -> None:
    """Save model, optimizer, and iteration state"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    src: str | os.PathLike | IO[bytes],
) -> int:
    """Load model, optimizer, and iteration state"""
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Construct BPE tokenizer"""

    class Tokenizer:
        def __init__(self, vocab, merges, special_tokens):
            self.vocab = vocab
            self.vocab_reverse = {v: k for k, v in vocab.items()}
            self.merges = merges
            self.special_tokens = special_tokens or []
            self.merge_priority = {(a, b): i for i, (a, b) in enumerate(merges)}
            # 特殊 token 按长度降序排列（长的优先匹配）
            self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

        def encode(self, text: str) -> list[int]:
            tokens = []
            # 按特殊 token 分割文本（长的优先匹配）
            parts = self._split_by_special_tokens(text)
            for part in parts:
                if part in self.special_tokens:
                    tokens.append(self.vocab_reverse[part.encode('utf-8')])
                else:
                    tokens.extend(self._bpe_encode(part))
            return tokens

        def _split_by_special_tokens(self, text: str) -> list[str]:
            """按特殊 token 分割文本，保留特殊 token 作为独立元素"""
            if not self.special_tokens:
                return [text]
            parts = [text]
            for st in self.special_tokens_sorted:
                new_parts = []
                for part in parts:
                    if part in self.special_tokens:
                        new_parts.append(part)
                    else:
                        segments = part.split(st)
                        for i, seg in enumerate(segments):
                            if i > 0:
                                new_parts.append(st)
                            if seg:
                                new_parts.append(seg)
                parts = new_parts
            return parts

        def _bpe_encode(self, text: str) -> list[int]:
            symbols = [bytes([b]) for b in text.encode('utf-8')]

            while len(symbols) > 1:
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
                    break

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

            return [self.vocab_reverse[s] for s in symbols]

        def decode(self, ids: list[int]) -> str:
            result = b''
            for token_id in ids:
                result += self.vocab[token_id]
            return result.decode('utf-8', errors='replace')

        def encode_iterable(self, iterable):
            """逐行编码，内存高效"""
            for line in iterable:
                yield from self.encode(line)

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer with optimized incremental pair counting"""
    from collections import Counter, defaultdict

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 初始化词表
    vocab = {}
    token_id = 0
    for st in special_tokens:
        vocab[token_id] = st.encode('utf-8')
        token_id += 1
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    # 按行处理，每行独立
    word_freq = Counter()
    for line in text.split('\n'):
        if not line:
            continue
        word_freq[tuple(bytes([b]) for b in line.encode('utf-8'))] += 1

    # 初始化: pair -> {word_idx: count_in_word}
    pair_in_word = defaultdict(lambda: defaultdict(int))
    pair_counts = Counter()
    word_list = list(word_freq.keys())
    word_freqs = [word_freq[w] for w in word_list]

    for idx, word in enumerate(word_list):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_in_word[pair][idx] += 1
            pair_counts[pair] += word_freqs[idx]

    merges = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # 找最高频 pair
        best_pair = max(pair_counts, key=pair_counts.get)
        new_token = best_pair[0] + best_pair[1]
        bp0, bp1 = best_pair

        # 获取受影响的 word indices
        affected = pair_in_word.pop(best_pair, {})
        if not affected:
            break

        for idx, count_in_word in affected.items():
            word = word_list[idx]
            freq = word_freqs[idx]

            # 移除旧 pair 计数
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq
                pair_in_word[pair][idx] -= 1
                if pair_in_word[pair][idx] <= 0:
                    del pair_in_word[pair][idx]

            # 执行合并
            new_word = []
            i = 0
            wlen = len(word)
            while i < wlen:
                if i < wlen - 1 and word[i] == bp0 and word[i + 1] == bp1:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word_list[idx] = new_word

            # 添加新 pair 计数
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counts[pair] += freq
                pair_in_word[pair][idx] += 1

        # 清理零计数
        pair_counts = +pair_counts

        vocab[token_id] = new_token
        merges.append(best_pair)
        token_id += 1

    return vocab, merges
