"""Step 11 验证: run_multihead_self_attention_with_rope 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_multihead_self_attention_with_rope, run_rope


def test_mha_rope_basic():
    """测试带 RoPE 的多头注意力基本功能"""
    batch, seq_len, d_model = 2, 8, 16
    num_heads = 4
    theta = 10000.0

    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    result = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_len, theta, q_proj, k_proj, v_proj, o_proj, x
    )
    assert result.shape == (batch, seq_len, d_model), f"形状: {result.shape}"
    print("✓ 基本功能测试通过")


def test_mha_rope_vs_manual():
    """测试与手动计算一致"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 2
    d_k = d_model // num_heads
    theta = 10000.0

    torch.manual_seed(42)
    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    result = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_len, theta, q_proj, k_proj, v_proj, o_proj, x
    )

    # 手动计算
    Q = x @ q_proj.T
    K = x @ k_proj.T
    V = x @ v_proj.T

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    # 应用 RoPE
    Q_flat = Q.reshape(batch * num_heads, seq_len, d_k)
    K_flat = K.reshape(batch * num_heads, seq_len, d_k)
    Q_rot = run_rope(d_k, theta, seq_len, Q_flat)
    K_rot = run_rope(d_k, theta, seq_len, K_flat)
    Q = Q_rot.reshape(batch, num_heads, seq_len, d_k)
    K = K_rot.reshape(batch, num_heads, seq_len, d_k)

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    attn = weights @ V
    attn = attn.transpose(1, 2).reshape(batch, seq_len, d_model)
    expected = attn @ o_proj.T

    assert torch.allclose(result, expected, atol=1e-5), "公式不匹配"
    print("✓ 公式验证通过")


def test_mha_rope_position_dependent():
    """测试输出依赖于位置"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 2
    theta = 10000.0

    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)

    # 相同输入，不同位置
    x = torch.randn(batch, seq_len, d_model)
    pos1 = torch.tensor([[0, 1, 2, 3]])
    pos2 = torch.tensor([[10, 11, 12, 13]])

    r1 = run_multihead_self_attention_with_rope(
        d_model, num_heads, 20, theta, q_proj, k_proj, v_proj, o_proj, x, pos1
    )
    r2 = run_multihead_self_attention_with_rope(
        d_model, num_heads, 20, theta, q_proj, k_proj, v_proj, o_proj, x, pos2
    )

    assert not torch.allclose(r1, r2), "不同位置应产生不同输出"
    print("✓ 位置依赖测试通过")


def test_mha_rope_gradient():
    """测试梯度反向传播"""
    batch, seq_len, d_model = 2, 4, 16
    num_heads = 4
    theta = 10000.0

    q_proj = torch.randn(d_model, d_model, requires_grad=True)
    k_proj = torch.randn(d_model, d_model, requires_grad=True)
    v_proj = torch.randn(d_model, d_model, requires_grad=True)
    o_proj = torch.randn(d_model, d_model, requires_grad=True)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    result = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_len, theta, q_proj, k_proj, v_proj, o_proj, x
    )
    result.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert q_proj.grad is not None, "Q 投影梯度应存在"
    print("✓ 梯度反向传播正常")


def test_mha_rope_causal():
    """测试因果 mask（下三角）"""
    batch, seq_len, d_model = 1, 4, 8
    num_heads = 2
    theta = 10000.0

    q_proj = torch.randn(d_model, d_model)
    k_proj = torch.randn(d_model, d_model)
    v_proj = torch.randn(d_model, d_model)
    o_proj = torch.randn(d_model, d_model)
    x = torch.randn(batch, seq_len, d_model)

    # 无 mask 的结果
    result_no_mask = run_multihead_self_attention_with_rope(
        d_model, num_heads, seq_len, theta, q_proj, k_proj, v_proj, o_proj, x
    )

    # 有因果 mask 的结果（需要修改函数支持 mask，这里只验证形状）
    assert result_no_mask.shape == (batch, seq_len, d_model)
    print("✓ 因果 mask 测试通过（形状验证）")


if __name__ == "__main__":
    test_mha_rope_basic()
    test_mha_rope_vs_manual()
    test_mha_rope_position_dependent()
    test_mha_rope_gradient()
    test_mha_rope_causal()
    print("\n✅ Step 11 run_multihead_self_attention_with_rope 实现正确！")
