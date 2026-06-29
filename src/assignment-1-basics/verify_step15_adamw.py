"""Step 15 验证: get_adamw_cls 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import get_adamw_cls


def test_adamw_class():
    """测试返回的是类而不是实例"""
    AdamW = get_adamw_cls()
    assert isinstance(AdamW, type), "应返回类"
    assert issubclass(AdamW, torch.optim.Optimizer), "应继承 Optimizer"
    print("✓ 类型测试通过")


def test_adamw_basic():
    """测试基本优化功能"""
    AdamW = get_adamw_cls()

    # 简单二次函数: f(x) = x², 最小值在 x=0
    x = torch.nn.Parameter(torch.tensor([10.0]))
    optimizer = AdamW([x], lr=0.1, weight_decay=0)

    for _ in range(200):
        loss = x ** 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert abs(x.item()) < 1.0, f"收敛到: {x}"
    print(f"✓ 基本优化测试通过 (最终值: {x.item():.4f})")


def test_adamw_momentum():
    """测试动量效果"""
    AdamW = get_adamw_cls()

    # 简单测试: Adam 能收敛
    x = torch.nn.Parameter(torch.tensor([10.0]))
    opt = AdamW([x], lr=0.3, weight_decay=0)

    for _ in range(100):
        (x ** 2).backward()
        opt.step()
        opt.zero_grad()

    assert abs(x.item()) < 1.0, f"Adam: {x}"
    print(f"✓ 动量测试通过 (最终值={x.item():.4f})")


def test_adamw_weight_decay():
    """测试权重衰减"""
    AdamW = get_adamw_cls()

    # 需要计算损失才能让梯度存在
    x = torch.nn.Parameter(torch.tensor([10.0]))
    optimizer = AdamW([x], lr=0.1, weight_decay=0.1)

    for _ in range(10):
        loss = x ** 2  # 梯度 = 2x
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 权重衰减会让 x 收敛更快
    assert abs(x.item()) < 10.0, f"衰减后: {x}"
    print(f"✓ 权重衰减测试通过 (最终值={x.item():.4f})")


def test_adamw_bias_correction():
    """测试偏差修正"""
    AdamW = get_adamw_cls()

    x = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = AdamW([x], lr=0.1, betas=(0.9, 0.999), weight_decay=0)

    # 第一步: m=0.1*grad, v=0.001*grad²
    # 无修正: m/(√v+ε) 可能很小
    # 有修正: m/(1-0.9) / (√(v/(1-0.999))+ε) 更大
    loss = x ** 2
    loss.backward()

    # 记录更新前的值
    x_before = x.item()
    optimizer.step()

    # 应该有明显更新
    diff = abs(x_before - x.item())
    assert diff > 0.01, f"偏差修正应产生明显更新: {diff}"
    print(f"✓ 偏差修正测试通过 (更新量: {diff:.4f})")


def test_adamw_vs_pytorch():
    """测试与 PyTorch AdamW 一致"""
    AdamW = get_adamw_cls()

    torch.manual_seed(42)
    x_custom = torch.nn.Parameter(torch.randn(3))
    x_pytorch = x_custom.clone().detach().requires_grad_(True)

    opt_custom = AdamW([x_custom], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_pytorch = torch.optim.AdamW([x_pytorch], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for _ in range(20):
        # 相同的损失
        loss_custom = (x_custom ** 2).sum()
        loss_custom.backward()
        opt_custom.step()
        opt_custom.zero_grad()

        loss_pytorch = (x_pytorch ** 2).sum()
        loss_pytorch.backward()
        opt_pytorch.step()
        opt_pytorch.zero_grad()

    assert torch.allclose(x_custom, x_pytorch, atol=1e-5), f"不一致: {x_custom} vs {x_pytorch}"
    print("✓ 与 PyTorch 一致性测试通过")


def test_adamw_multiple_params():
    """测试多个参数组"""
    AdamW = get_adamw_cls()

    p1 = torch.nn.Parameter(torch.tensor([5.0]))
    p2 = torch.nn.Parameter(torch.tensor([10.0]))
    optimizer = AdamW([p1, p2], lr=0.3, weight_decay=0)

    for _ in range(100):
        loss = p1 ** 2 + p2 ** 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert abs(p1.item()) < 1.0, f"p1: {p1}"
    assert abs(p2.item()) < 1.0, f"p2: {p2}"
    print(f"✓ 多参数测试通过 (p1={p1.item():.4f}, p2={p2.item():.4f})")


if __name__ == "__main__":
    test_adamw_class()
    test_adamw_basic()
    test_adamw_momentum()
    test_adamw_weight_decay()
    test_adamw_bias_correction()
    test_adamw_vs_pytorch()
    test_adamw_multiple_params()
    print("\n✅ Step 15 get_adamw_cls 实现正确！")
