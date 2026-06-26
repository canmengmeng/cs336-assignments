"""Step 2 验证: run_softmax 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_softmax


def test_softmax_basic():
    """测试 Softmax 基本性质"""
    # 所有元素相同时，softmax 应均匀分布
    x = torch.tensor([[5.0, 5.0, 5.0]])
    result = run_softmax(x, dim=-1)
    expected = torch.tensor([[1/3, 1/3, 1/3]])
    assert torch.allclose(result, expected, atol=1e-6), f"均匀分布: {result}"
    print("✓ 均匀分布测试通过")

    # 最大值应接近 1，其余接近 0
    x = torch.tensor([[10.0, 0.0, 0.0]])
    result = run_softmax(x, dim=-1)
    assert result[0, 0] > 0.99, f"最大值应接近 1: {result[0, 0]}"
    print("✓ 极端值测试通过")


def test_softmax_sum():
    """测试 softmax 输出之和为 1"""
    x = torch.randn(3, 5)
    result = run_softmax(x, dim=-1)
    sums = result.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-6), f"和不为 1: {sums}"
    print("✓ 求和为 1 测试通过")


def test_softmax_numerical_stability():
    """测试数值稳定性（大数值不溢出）"""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    result = run_softmax(x, dim=-1)
    assert not torch.isnan(result).any(), "不应出现 NaN"
    assert not torch.isinf(result).any(), "不应出现 Inf"
    assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-6)
    print("✓ 数值稳定性测试通过")


def test_softmax_different_dims():
    """测试不同维度的 softmax"""
    x = torch.randn(2, 3, 4)

    # dim=0
    result = run_softmax(x, dim=0)
    sums = result.sum(dim=0)
    assert torch.allclose(sums, torch.ones(3, 4), atol=1e-6)

    # dim=1
    result = run_softmax(x, dim=1)
    sums = result.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2, 4), atol=1e-6)

    # dim=2
    result = run_softmax(x, dim=2)
    sums = result.sum(dim=2)
    assert torch.allclose(sums, torch.ones(2, 3), atol=1e-6)

    print("✓ 多维度测试通过")


def test_softmax_gradient():
    """测试梯度可以反向传播"""
    x = torch.randn(2, 3, requires_grad=True)
    y = run_softmax(x, dim=-1)
    y.sum().backward()
    assert x.grad is not None, "梯度应存在"
    print("✓ 梯度反向传播正常")


def test_softmax_vs_pytorch():
    """测试与 PyTorch 官方实现一致"""
    x = torch.randn(4, 5)
    expected = torch.softmax(x, dim=-1)
    result = run_softmax(x, dim=-1)
    assert torch.allclose(result, expected, atol=1e-6), "与 PyTorch 实现不一致"
    print("✓ 与 PyTorch 一致性测试通过")


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax_sum()
    test_softmax_numerical_stability()
    test_softmax_different_dims()
    test_softmax_gradient()
    test_softmax_vs_pytorch()
    print("\n✅ Step 2 run_softmax 实现正确！")
