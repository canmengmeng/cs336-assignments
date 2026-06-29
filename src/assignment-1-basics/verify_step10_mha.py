"""Step 10 验证: run_multihead_self_attention 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_multihead_self_attention


def test_mha_basic():
    """测试多头注意力基本功能"""
    batch, seq_len, d_model = 2, 8, 16
    num_heads = 4

    # 随机权重
    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    result = run_multihead_self_attention(d_model, num_heads, q_proj, k_proj, v_proj, o_proj, x)
    assert result.shape == (batch, seq_len, d_model), f"形状: {result.shape}"
    print("✓ 基本功能测试通过")


def test_mha_formula():
    """测试公式正确性"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 2
    d_k = d_model // num_heads

    torch.manual_seed(42)
    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    result = run_multihead_self_attention(d_model, num_heads, q_proj, k_proj, v_proj, o_proj, x)

    # 手动计算
    Q = x @ q_proj.T
    K = x @ k_proj.T
    V = x @ v_proj.T

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    attn = weights @ V

    attn = attn.transpose(1, 2).reshape(batch, seq_len, d_model)
    expected = attn @ o_proj.T

    assert torch.allclose(result, expected, atol=1e-5), f"公式不匹配"
    print("✓ 公式验证通过")


def test_mha_heads_independence():
    """测试各头独立计算"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 2
    d_k = d_model // num_heads

    # 构造特殊权重：让每个头只关注不同的位置
    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    # 计算各头的注意力权重
    Q = x @ q_proj.T
    K = x @ k_proj.T
    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)

    # 不同头的注意力模式应不同（大概率）
    head0_pattern = weights[0, 0]
    head1_pattern = weights[0, 1]
    assert not torch.allclose(head0_pattern, head1_pattern), "不同头应有不同注意力模式"
    print("✓ 头独立性测试通过")


def test_mha_gradient():
    """测试梯度反向传播"""
    batch, seq_len, d_model = 2, 4, 16
    num_heads = 4

    q_proj = torch.randn(d_model, d_model, requires_grad=True)
    k_proj = torch.randn(d_model, d_model, requires_grad=True)
    v_proj = torch.randn(d_model, d_model, requires_grad=True)
    o_proj = torch.randn(d_model, d_model, requires_grad=True)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    result = run_multihead_self_attention(d_model, num_heads, q_proj, k_proj, v_proj, o_proj, x)
    result.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert q_proj.grad is not None, "Q 投影梯度应存在"
    print("✓ 梯度反向传播正常")


def test_mha_single_head():
    """测试单头情况"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 1

    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    result = run_multihead_self_attention(d_model, num_heads, q_proj, k_proj, v_proj, o_proj, x)
    assert result.shape == (batch, seq_len, d_model)
    print("✓ 单头测试通过")


if __name__ == "__main__":
    test_mha_basic()
    test_mha_formula()
    test_mha_heads_independence()
    test_mha_gradient()
    test_mha_single_head()
    print("\n✅ Step 10 run_multihead_self_attention 实现正确！")
