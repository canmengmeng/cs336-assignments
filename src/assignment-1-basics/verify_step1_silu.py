"""Step 1 验证: run_silu 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_silu


def test_silu_basic():
    """测试 SiLU 基本性质"""
    # SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    x = torch.tensor([0.0])
    assert torch.allclose(run_silu(x), torch.tensor([0.0])), "SiLU(0) 应为 0"

    # SiLU(很大) ≈ x (因为 sigmoid(很大) ≈ 1)
    x = torch.tensor([10.0])
    assert torch.allclose(run_silu(x), x, atol=1e-3), "SiLU(大正数) ≈ x"

    # SiLU(很小) ≈ 0 (因为 sigmoid(很小) ≈ 0)
    x = torch.tensor([-10.0])
    assert torch.allclose(run_silu(x), torch.tensor([0.0]), atol=1e-3), "SiLU(大负数) ≈ 0"

    print("✓ 基本性质测试通过")


def test_silu_formula():
    """测试 SiLU(x) = x * sigmoid(x)"""
    x = torch.randn(3, 4)
    expected = x * torch.sigmoid(x)
    result = run_silu(x)
    assert torch.allclose(result, expected), "SiLU 公式不匹配"
    print("✓ 公式验证通过")


def test_silu_gradient():
    """测试梯度可以反向传播"""
    x = torch.randn(2, 3, requires_grad=True)
    y = run_silu(x)
    y.sum().backward()
    assert x.grad is not None, "梯度应存在"
    print("✓ 梯度反向传播正常")


def test_silu_shapes():
    """测试各种输入形状"""
    shapes = [(5,), (2, 3), (2, 3, 4), (1, 2, 3, 4)]
    for shape in shapes:
        x = torch.randn(shape)
        y = run_silu(x)
        assert y.shape == shape, f"形状 {shape} 不匹配"
    print("✓ 多形状测试通过")


if __name__ == "__main__":
    test_silu_basic()
    test_silu_formula()
    test_silu_gradient()
    test_silu_shapes()
    print("\n✅ Step 1 run_silu 实现正确！")
