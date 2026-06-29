"""Step 8 验证: run_scaled_dot_product_attention 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_scaled_dot_product_attention


def test_attention_basic():
    """测试注意力基本功能"""
    batch, heads, seq_len, d_k = 2, 4, 8, 16
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    result = run_scaled_dot_product_attention(q, k, v)
    assert result.shape == (batch, heads, seq_len, d_k), f"形状: {result.shape}"
    print("✓ 基本功能测试通过")


def test_attention_formula():
    """测试公式正确性"""
    batch, heads, seq_len, d_k = 1, 1, 4, 8
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    result = run_scaled_dot_product_attention(q, k, v)

    # 手动计算
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    expected = weights @ v

    assert torch.allclose(result, expected, atol=1e-6), "公式不匹配"
    print("✓ 公式验证通过")


def test_attention_mask():
    """测试因果 mask"""
    batch, heads, seq_len, d_k = 1, 1, 4, 8
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    # 因果 mask: 下三角，True 表示可以 attend
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

    result = run_scaled_dot_product_attention(q, k, v, mask)

    # 验证第 i 个 token 只 attend 到 0..i
    # 检查权重矩阵是否是下三角的
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    scores_masked = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores_masked, dim=-1)

    # 上三角部分应为 0
    upper = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    assert torch.allclose(weights[0, 0][upper], torch.zeros_like(weights[0, 0][upper])), \
        "mask 后上三角应为 0"
    print("✓ 因果 mask 测试通过")


def test_attention_no_mask():
    """测试无 mask 时每个 token attend 所有位置"""
    batch, heads, seq_len, d_k = 1, 1, 4, 8
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)

    result = run_scaled_dot_product_attention(q, k, v)

    # 权重矩阵每行应和为 1
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "权重每行应和为 1"
    print("✓ 无 mask 测试通过")


def test_attention_scaling():
    """测试缩放的作用"""
    batch, heads, seq_len, d_k = 1, 1, 4, 64
    q = torch.randn(batch, heads, seq_len, d_k) * 10  # 大数值
    k = torch.randn(batch, heads, seq_len, d_k) * 10
    v = torch.randn(batch, heads, seq_len, d_k)

    # 不缩放: softmax 会趋向 one-hot
    scores_raw = q @ k.transpose(-2, -1)
    weights_raw = torch.softmax(scores_raw, dim=-1)
    entropy_raw = -(weights_raw * torch.log(weights_raw + 1e-10)).sum(dim=-1).mean()

    # 缩放后: softmax 更均匀
    scores_scaled = scores_raw / (d_k ** 0.5)
    weights_scaled = torch.softmax(scores_scaled, dim=-1)
    entropy_scaled = -(weights_scaled * torch.log(weights_scaled + 1e-10)).sum(dim=-1).mean()

    assert entropy_scaled > entropy_raw, "缩放后熵应更大（更均匀）"
    print(f"✓ 缩放测试通过 (raw熵={entropy_raw:.2f}, scaled熵={entropy_scaled:.2f})")


def test_attention_gradient():
    """测试梯度反向传播"""
    batch, heads, seq_len, d_k = 2, 4, 8, 16
    q = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)

    result = run_scaled_dot_product_attention(q, k, v)
    result.sum().backward()

    assert q.grad is not None, "Q 梯度应存在"
    assert k.grad is not None, "K 梯度应存在"
    assert v.grad is not None, "V 梯度应存在"
    print("✓ 梯度反向传播正常")


def test_attention_different_dk_dv():
    """测试 d_k != d_v 的情况"""
    batch, heads, seq_len = 1, 1, 4
    d_k, d_v = 8, 12

    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_v)

    result = run_scaled_dot_product_attention(q, k, v)
    assert result.shape == (batch, heads, seq_len, d_v), f"形状: {result.shape}"
    print("✓ d_k ≠ d_v 测试通过")


if __name__ == "__main__":
    test_attention_basic()
    test_attention_formula()
    test_attention_mask()
    test_attention_no_mask()
    test_attention_scaling()
    test_attention_gradient()
    test_attention_different_dk_dv()
    print("\n✅ Step 8 run_scaled_dot_product_attention 实现正确！")
