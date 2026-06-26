"""补充验证: run_swiglu 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_swiglu, run_silu


def test_swiglu_basic():
    """测试 SwiGLU 基本功能"""
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    w3 = torch.randn(d_ff, d_model)
    x = torch.randn(2, d_model)

    result = run_swiglu(d_model, d_ff, w1, w2, w3, x)
    assert result.shape == (2, d_model), f"形状错误: {result.shape}"
    print("✓ 基本功能测试通过")


def test_swiglu_formula():
    """测试公式正确性"""
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    w3 = torch.randn(d_ff, d_model)
    x = torch.randn(3, d_model)

    result = run_swiglu(d_model, d_ff, w1, w2, w3, x)

    # 手动计算
    gate = run_silu(x @ w1.T)      # (*, d_ff)
    up = x @ w3.T                   # (*, d_ff)
    gated = gate * up               # (*, d_ff)
    expected = gated @ w2.T         # (*, d_model)

    assert torch.allclose(result, expected, atol=1e-6), f"公式不匹配"
    print("✓ 公式验证通过")


def test_swiglu_gating():
    """测试门控机制"""
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    w3 = torch.randn(d_ff, d_model)
    x = torch.randn(2, d_model)

    # Gate 路径为 0 时，输出应为 0
    w1_zero = torch.zeros_like(w1)
    result = run_swiglu(d_model, d_ff, w1_zero, w2, w3, x)
    assert torch.allclose(result, torch.zeros_like(result)), "gate=0 时输出应为 0"
    print("✓ 门控机制测试通过")


def test_swiglu_gradient():
    """测试梯度反向传播"""
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model, requires_grad=True)
    w2 = torch.randn(d_model, d_ff, requires_grad=True)
    w3 = torch.randn(d_ff, d_model, requires_grad=True)
    x = torch.randn(2, d_model, requires_grad=True)

    y = run_swiglu(d_model, d_ff, w1, w2, w3, x)
    y.sum().backward()

    assert x.grad is not None, "输入梯度应存在"
    assert w1.grad is not None, "W1 梯度应存在"
    assert w2.grad is not None, "W2 梯度应存在"
    assert w3.grad is not None, "W3 梯度应存在"
    print("✓ 梯度反向传播正常")


def test_swiglu_batch_shapes():
    """测试各种 batch 形状"""
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    w3 = torch.randn(d_ff, d_model)

    shapes = [(d_model,), (2, d_model), (2, 5, d_model)]

    for shape in shapes:
        x = torch.randn(shape)
        result = run_swiglu(d_model, d_ff, w1, w2, w3, x)
        expected_shape = (*shape[:-1], d_model)
        assert result.shape == expected_shape, f"形状 {shape}: 期望 {expected_shape}"

    print("✓ 多形状测试通过")


if __name__ == "__main__":
    test_swiglu_basic()
    test_swiglu_formula()
    test_swiglu_gating()
    test_swiglu_gradient()
    test_swiglu_batch_shapes()
    print("\n✅ run_swiglu 实现正确！")
