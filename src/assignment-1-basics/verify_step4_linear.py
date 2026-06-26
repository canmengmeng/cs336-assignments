"""Step 4 验证: run_linear 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_linear


def test_linear_basic():
    """测试线性变换基本功能"""
    d_in, d_out = 4, 3
    weights = torch.randn(d_out, d_in)
    x = torch.randn(2, d_in)

    result = run_linear(d_in, d_out, weights, x)
    expected = x @ weights.T

    assert torch.allclose(result, expected), "结果不匹配"
    print("✓ 基本线性变换测试通过")


def test_linear_identity():
    """测试恒等变换（单位矩阵权重）"""
    d = 4
    # 单位矩阵作为权重: y = x @ I^T = x
    weights = torch.eye(d)
    x = torch.randn(2, d)

    result = run_linear(d, d, weights, x)
    assert torch.allclose(result, x, atol=1e-6), "恒等变换应输出输入"
    print("✓ 恒等变换测试通过")


def test_linear_batch_shapes():
    """测试各种 batch 形状"""
    d_in, d_out = 4, 3
    weights = torch.randn(d_out, d_in)

    shapes = [
        (d_in,),              # 单向量
        (2, d_in),            # batch=2
        (2, 5, d_in),         # batch=2, seq_len=5
        (2, 5, 10, d_in),     # 4D
    ]

    for shape in shapes:
        x = torch.randn(shape)
        result = run_linear(d_in, d_out, weights, x)
        expected_shape = (*shape[:-1], d_out)
        assert result.shape == expected_shape, f"形状 {shape}: 期望 {expected_shape}, 得到 {result.shape}"

    print("✓ 多形状测试通过")


def test_linear_gradient():
    """测试梯度反向传播"""
    d_in, d_out = 4, 3
    weights = torch.randn(d_out, d_in, requires_grad=True)
    x = torch.randn(2, d_in, requires_grad=True)

    y = run_linear(d_in, d_out, weights, x)
    y.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert weights.grad is not None, "权重梯度应存在"
    print("✓ 梯度反向传播正常")


def test_linear_vs_pytorch():
    """测试与 PyTorch nn.Linear 一致"""
    d_in, d_out = 4, 3
    weights = torch.randn(d_out, d_in)
    bias = torch.zeros(d_out)

    layer = torch.nn.Linear(d_in, d_out)
    layer.weight.data = weights.clone()
    layer.bias.data = bias.clone()

    x = torch.randn(2, d_in)
    expected = layer(x)
    result = run_linear(d_in, d_out, weights, x)

    assert torch.allclose(result, expected, atol=1e-6), "与 PyTorch nn.Linear 不一致"
    print("✓ 与 PyTorch 一致性测试通过")


if __name__ == "__main__":
    test_linear_basic()
    test_linear_identity()
    test_linear_batch_shapes()
    test_linear_gradient()
    test_linear_vs_pytorch()
    print("\n✅ Step 4 run_linear 实现正确！")
