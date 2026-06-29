"""Step 14 验证: run_gradient_clipping 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_gradient_clipping


def test_clipping_basic():
    """测试基本裁剪功能"""
    # 创建参数，设置梯度
    p1 = torch.randn(3, 4)
    p1.grad = torch.tensor([[3.0, 4.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])

    p2 = torch.randn(2)
    p2.grad = torch.tensor([1.0, 0.0])

    # 总范数 = √(25 + 1) = √26 ≈ 5.099
    run_gradient_clipping([p1, p2], max_l2_norm=1.0)

    # 验证裁剪后总范数 ≈ 1.0
    total_norm = torch.sqrt(torch.sum(p1.grad ** 2) + torch.sum(p2.grad ** 2))
    assert torch.allclose(total_norm, torch.tensor(1.0), atol=1e-5), f"总范数: {total_norm}"
    print("✓ 基本裁剪测试通过")


def test_clipping_no_clip():
    """测试梯度范数小于阈值时不裁剪"""
    p = torch.randn(3)
    original_grad = torch.tensor([0.1, 0.2, 0.3])
    p.grad = original_grad.clone()

    run_gradient_clipping([p], max_l2_norm=10.0)

    # 梯度应保持不变
    assert torch.allclose(p.grad, original_grad), "小于阈值不应裁剪"
    print("✓ 无需裁剪测试通过")


def test_clipping_preserves_direction():
    """测试裁剪保持梯度方向"""
    p = torch.randn(3)
    p.grad = torch.tensor([3.0, 4.0, 0.0])
    original_direction = p.grad / torch.norm(p.grad)

    run_gradient_clipping([p], max_l2_norm=1.0)

    new_direction = p.grad / torch.norm(p.grad)
    assert torch.allclose(original_direction, new_direction, atol=1e-5), "方向应保持不变"
    print("✓ 方向保持测试通过")


def test_clipping_inplace():
    """测试原地修改"""
    p = torch.randn(3)
    p.grad = torch.tensor([10.0, 0.0, 0.0])
    grad_ref = p.grad

    run_gradient_clipping([p], max_l2_norm=1.0)

    # 应该是原地修改，不是创建新张量
    assert p.grad is grad_ref, "应原地修改"
    print("✓ 原地修改测试通过")


def test_clipping_single_parameter():
    """测试单个参数输入"""
    p = torch.nn.Parameter(torch.randn(3))
    p.grad = torch.tensor([100.0, 0.0, 0.0])

    run_gradient_clipping(p, max_l2_norm=1.0)

    total_norm = torch.norm(p.grad)
    assert torch.allclose(total_norm, torch.tensor(1.0), atol=1e-5)
    print("✓ 单参数测试通过")


def test_clipping_no_grad():
    """测试没有梯度的参数"""
    p1 = torch.randn(3)
    p1.grad = torch.tensor([10.0, 0.0, 0.0])

    p2 = torch.randn(3)
    p2.grad = None  # 无梯度

    # 不应报错
    run_gradient_clipping([p1, p2], max_l2_norm=1.0)
    assert torch.allclose(torch.norm(p1.grad), torch.tensor(1.0), atol=1e-5)
    print("✓ 无梯度参数测试通过")


def test_clipping_exact_norm():
    """测试梯度范数恰好等于阈值"""
    p = torch.randn(3)
    p.grad = torch.tensor([0.6, 0.8, 0.0])  # 范数 = 1.0
    original = p.grad.clone()

    run_gradient_clipping([p], max_l2_norm=1.0)

    # 应保持不变
    assert torch.allclose(p.grad, original), "恰好等于阈值不应修改"
    print("✓ 精确范数测试通过")


if __name__ == "__main__":
    test_clipping_basic()
    test_clipping_no_clip()
    test_clipping_preserves_direction()
    test_clipping_inplace()
    test_clipping_single_parameter()
    test_clipping_no_grad()
    test_clipping_exact_norm()
    print("\n✅ Step 14 run_gradient_clipping 实现正确！")
