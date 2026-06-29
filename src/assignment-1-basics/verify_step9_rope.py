"""Step 9 验证: run_rope 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_rope


def test_rope_basic():
    """测试 RoPE 基本功能"""
    batch, seq_len, d_model = 2, 8, 16
    theta = 10000.0
    x = torch.randn(batch, seq_len, d_model)

    result = run_rope(d_model, theta, seq_len, x)
    assert result.shape == x.shape, f"形状不匹配: {result.shape}"
    print("✓ 基本功能测试通过")


def test_rope_preserves_norm():
    """测试旋转不改变向量长度（正交变换）"""
    batch, seq_len, d_model = 1, 4, 16
    theta = 10000.0
    x = torch.randn(batch, seq_len, d_model)

    result = run_rope(d_model, theta, seq_len, x)

    # 旋转是正交变换，保持范数
    norm_before = torch.norm(x, dim=-1)
    norm_after = torch.norm(result, dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5), \
        f"范数不匹配: {norm_before} vs {norm_after}"
    print("✓ 范数保持测试通过")


def test_rope_position_0():
    """测试位置 0 的旋转应接近恒等变换"""
    batch, seq_len, d_model = 1, 4, 16
    theta = 10000.0
    x = torch.randn(batch, seq_len, d_model)

    # 只旋转位置 0 的 token
    positions = torch.zeros(batch, 1, dtype=torch.long)
    x_single = x[:, 0:1, :]
    result = run_rope(d_model, theta, seq_len, x_single, positions)

    # cos(0)=1, sin(0)=0, 所以旋转后应等于原向量
    assert torch.allclose(result, x_single, atol=1e-5), "位置 0 应保持不变"
    print("✓ 位置 0 恒等变换测试通过")


def test_rope_relative_position():
    """测试相对位置编码特性"""
    batch, seq_len, d_model = 1, 1, 16
    theta = 10000.0
    x = torch.randn(batch, 1, d_model)

    # 位置 m 和位置 n 的旋转
    pos_m = torch.tensor([[5]])
    pos_n = torch.tensor([[8]])

    q_rotated = run_rope(d_model, theta, 10, x.clone(), pos_m)
    k_rotated = run_rope(d_model, theta, 10, x.clone(), pos_n)

    # Q·K 的点积应只依赖相对位置 (5-8 = -3)
    dot_product = (q_rotated * k_rotated).sum()

    # 用另一对相同相对位置的验证
    pos_m2 = torch.tensor([[10]])
    pos_n2 = torch.tensor([[13]])
    q_rotated2 = run_rope(d_model, theta, 20, x.clone(), pos_m2)
    k_rotated2 = run_rope(d_model, theta, 20, x.clone(), pos_n2)
    dot_product2 = (q_rotated2 * k_rotated2).sum()

    assert torch.allclose(dot_product, dot_product2, atol=1e-4), \
        f"相同相对位置应有相同点积: {dot_product} vs {dot_product2}"
    print("✓ 相对位置编码测试通过")


def test_rope_custom_positions():
    """测试自定义位置参数"""
    batch, seq_len, d_model = 2, 4, 16
    theta = 10000.0
    x = torch.randn(batch, seq_len, d_model)
    # positions 形状必须和 (batch, seq_len) 一致
    positions = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])

    result = run_rope(d_model, theta, 20, x, positions)
    assert result.shape == x.shape, f"形状不匹配: {result.shape}"
    print("✓ 自定义位置测试通过")


def test_rope_different_positions_different_output():
    """测试不同位置产生不同输出"""
    d_model = 16
    theta = 10000.0
    x = torch.randn(1, 1, d_model)

    pos_0 = torch.tensor([[0]])
    pos_1 = torch.tensor([[1]])

    r0 = run_rope(d_model, theta, 10, x.clone(), pos_0)
    r1 = run_rope(d_model, theta, 10, x.clone(), pos_1)

    assert not torch.allclose(r0, r1), "不同位置应产生不同输出"
    print("✓ 位置区分测试通过")


def test_rope_gradient():
    """测试梯度反向传播"""
    batch, seq_len, d_model = 2, 4, 16
    theta = 10000.0
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    result = run_rope(d_model, theta, seq_len, x)
    result.sum().backward()

    assert x.grad is not None, "梯度应存在"
    print("✓ 梯度反向传播正常")


if __name__ == "__main__":
    test_rope_basic()
    test_rope_preserves_norm()
    test_rope_position_0()
    test_rope_relative_position()
    test_rope_custom_positions()
    test_rope_different_positions_different_output()
    test_rope_gradient()
    print("\n✅ Step 9 run_rope 实现正确！")
