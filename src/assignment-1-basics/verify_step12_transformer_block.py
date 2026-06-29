"""Step 12 验证: run_transformer_block 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import (
    run_transformer_block,
    run_rmsnorm,
    run_multihead_self_attention_with_rope,
    run_swiglu,
)


def make_weights(d_model, num_heads, d_ff):
    """构造随机权重字典"""
    return {
        'attn.q_proj.weight': torch.randn(d_model, d_model),
        'attn.k_proj.weight': torch.randn(d_model, d_model),
        'attn.v_proj.weight': torch.randn(d_model, d_model),
        'attn.o_proj.weight': torch.randn(d_model, d_model),
        'ln1.weight': torch.randn(d_model),
        'ffn.w1.weight': torch.randn(d_ff, d_model),
        'ffn.w2.weight': torch.randn(d_model, d_ff),
        'ffn.w3.weight': torch.randn(d_ff, d_model),
        'ln2.weight': torch.randn(d_model),
    }


def test_transformer_block_basic():
    """测试基本功能"""
    batch, seq_len, d_model = 2, 8, 16
    num_heads, d_ff = 4, 64
    theta = 10000.0

    weights = make_weights(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)

    result = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)
    assert result.shape == (batch, seq_len, d_model), f"形状: {result.shape}"
    print("✓ 基本功能测试通过")


def test_transformer_block_formula():
    """测试公式正确性"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads, d_ff = 2, 32
    theta = 10000.0

    weights = make_weights(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)

    result = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)

    # 手动计算
    # 1. 注意力子层
    normed1 = run_rmsnorm(d_model, 1e-5, weights['ln1.weight'], x)
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_len, theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.o_proj.weight'],
        normed1
    )
    h = x + attn_out

    # 2. FFN 子层
    normed2 = run_rmsnorm(d_model, 1e-5, weights['ln2.weight'], h)
    ffn_out = run_swiglu(
        d_model, d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        normed2
    )
    expected = h + ffn_out

    assert torch.allclose(result, expected, atol=1e-5), "公式不匹配"
    print("✓ 公式验证通过")


def test_transformer_block_residual():
    """测试残差连接的作用"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads, d_ff = 2, 32
    theta = 10000.0

    weights = make_weights(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)

    result = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)

    # 残差连接意味着输出和输入在同一数量级
    input_norm = x.abs().mean()
    output_norm = result.abs().mean()
    ratio = output_norm / input_norm
    assert ratio < 100, f"残差连接应保持数量级，比值: {ratio}"
    print(f"✓ 残差连接测试通过 (输出/输入比值: {ratio:.2f})")


def test_transformer_block_gradient():
    """测试梯度反向传播"""
    batch, seq_len, d_model = 2, 4, 16
    num_heads, d_ff = 4, 64
    theta = 10000.0

    weights = make_weights(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    # 需要对权重也设置 requires_grad
    for k in weights:
        weights[k].requires_grad_(True)

    result = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)
    result.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert weights['ln1.weight'].grad is not None, "ln1 梯度应存在"
    assert weights['attn.q_proj.weight'].grad is not None, "Q 投影梯度应存在"
    print("✓ 梯度反向传播正常")


def test_transformer_block_deterministic():
    """测试确定性（相同输入相同输出）"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads, d_ff = 2, 32
    theta = 10000.0

    weights = make_weights(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)

    r1 = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)
    r2 = run_transformer_block(d_model, num_heads, d_ff, seq_len, theta, weights, x)

    assert torch.equal(r1, r2), "相同输入应产生相同输出"
    print("✓ 确定性测试通过")


if __name__ == "__main__":
    test_transformer_block_basic()
    test_transformer_block_formula()
    test_transformer_block_residual()
    test_transformer_block_gradient()
    test_transformer_block_deterministic()
    print("\n✅ Step 12 run_transformer_block 实现正确！")
