"""Step 3 验证: run_rmsnorm 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_rmsnorm


def test_rmsnorm_basic():
    """测试 RMSNorm 基本性质"""
    d_model = 4
    weights = torch.ones(d_model)
    eps = 1e-5

    # 全 1 向量: RMS = sqrt(mean([1,1,1,1])) = 1, 输出应为全 1
    x = torch.ones(1, d_model)
    result = run_rmsnorm(d_model, eps, weights, x)
    assert torch.allclose(result, x, atol=1e-5), f"全1输入: {result}"
    print("✓ 全 1 向量测试通过")

    # 验证归一化后 RMS ≈ 1
    x = torch.randn(2, d_model) * 10  # 大数值
    result = run_rmsnorm(d_model, eps, weights, x)
    rms_after = torch.sqrt(torch.mean(result ** 2, dim=-1))
    assert torch.allclose(rms_after, torch.ones(2), atol=1e-4), f"归一化后 RMS: {rms_after}"
    print("✓ 归一化后 RMS ≈ 1 测试通过")


def test_rmsnorm_formula():
    """测试公式正确性"""
    d_model = 3
    weights = torch.tensor([1.0, 2.0, 0.5])
    eps = 1e-5

    x = torch.tensor([[2.0, 4.0, 6.0]])
    result = run_rmsnorm(d_model, eps, weights, x)

    # 手动计算
    mean_sq = (4 + 16 + 36) / 3  # 18.6667
    rms = torch.sqrt(torch.tensor(mean_sq + eps))
    expected = x / rms * weights

    assert torch.allclose(result, expected, atol=1e-5), f"公式不匹配: {result} vs {expected}"
    print("✓ 公式验证通过")


def test_rmsnorm_weights():
    """测试可学习权重的作用"""
    d_model = 4
    eps = 1e-5

    x = torch.randn(2, d_model)

    # weights 全 1
    w1 = torch.ones(d_model)
    r1 = run_rmsnorm(d_model, eps, w1, x)

    # weights 全 2
    w2 = torch.ones(d_model) * 2
    r2 = run_rmsnorm(d_model, eps, w2, x)

    # r2 应该是 r1 的 2 倍
    assert torch.allclose(r2, r1 * 2, atol=1e-5), "权重缩放不正确"
    print("✓ 权重缩放测试通过")


def test_rmsnorm_no_centering():
    """测试 RMSNorm 不减均值（区别于 LayerNorm）"""
    d_model = 4
    weights = torch.ones(d_model)
    eps = 1e-5

    # 选择一个均值不为 0 的输入
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    result = run_rmsnorm(d_model, eps, weights, x)

    # 输出的均值不应为 0（LayerNorm 会是 0）
    assert result.mean() != 0, "RMSNorm 不应中心化"
    print("✓ 无中心化测试通过")


def test_rmsnorm_gradient():
    """测试梯度可以反向传播"""
    d_model = 4
    weights = torch.ones(d_model, requires_grad=True)
    x = torch.randn(2, d_model, requires_grad=True)

    y = run_rmsnorm(d_model, 1e-5, weights, x)
    y.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert weights.grad is not None, "权重梯度应存在"
    print("✓ 梯度反向传播正常")


def test_rmsnorm_numerical_stability():
    """测试数值稳定性"""
    d_model = 4
    weights = torch.ones(d_model)

    # 极小值
    x_small = torch.tensor([[1e-8, 1e-8, 1e-8, 1e-8]])
    result = run_rmsnorm(d_model, 1e-5, weights, x_small)
    assert not torch.isnan(result).any(), "极小值不应产生 NaN"

    # 极大值
    x_large = torch.tensor([[1e8, 1e8, 1e8, 1e8]])
    result = run_rmsnorm(d_model, 1e-5, weights, x_large)
    assert not torch.isnan(result).any(), "极大值不应产生 NaN"
    print("✓ 数值稳定性测试通过")


if __name__ == "__main__":
    test_rmsnorm_basic()
    test_rmsnorm_formula()
    test_rmsnorm_weights()
    test_rmsnorm_no_centering()
    test_rmsnorm_gradient()
    test_rmsnorm_numerical_stability()
    print("\n✅ Step 3 run_rmsnorm 实现正确！")
