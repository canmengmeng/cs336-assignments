"""Step 6 验证: run_cross_entropy 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_cross_entropy


def test_cross_entropy_basic():
    """测试交叉熵基本功能"""
    # 完美预测: logits 在正确位置很大
    inputs = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    loss = run_cross_entropy(inputs, targets)
    assert loss < 0.01, f"完美预测 loss 应很小: {loss}"
    print(f"✓ 完美预测 loss={loss:.4f}")

    # 糟糕预测: logits 在错误位置很大
    inputs = torch.tensor([[0.0, 10.0, 0.0]])
    targets = torch.tensor([0])
    loss = run_cross_entropy(inputs, targets)
    assert loss > 5.0, f"糟糕预测 loss 应很大: {loss}"
    print(f"✓ 糟糕预测 loss={loss:.4f}")


def test_cross_entropy_numerical_stability():
    """测试数值稳定性"""
    # 大数值
    inputs = torch.tensor([[1000.0, 1001.0, 1002.0]])
    targets = torch.tensor([2])
    loss = run_cross_entropy(inputs, targets)
    assert not torch.isnan(loss), "大数值不应产生 NaN"
    assert not torch.isinf(loss), "大数值不应产生 Inf"
    print(f"✓ 大数值稳定性 loss={loss:.4f}")

    # 小数值
    inputs = torch.tensor([[-1000.0, -1001.0, -1002.0]])
    targets = torch.tensor([0])
    loss = run_cross_entropy(inputs, targets)
    assert not torch.isnan(loss), "小数值不应产生 NaN"
    assert not torch.isinf(loss), "小数值不应产生 Inf"
    print(f"✓ 小数值稳定性 loss={loss:.4f}")


def test_cross_entropy_formula():
    """测试公式正确性"""
    inputs = torch.tensor([[2.0, 1.0, 0.1]])
    targets = torch.tensor([0])
    result = run_cross_entropy(inputs, targets)

    # 手动计算
    softmax = torch.softmax(inputs, dim=-1)
    expected = -torch.log(softmax[0, 0])

    assert torch.allclose(result, expected, atol=1e-6), f"公式不匹配: {result} vs {expected}"
    print("✓ 公式验证通过")


def test_cross_entropy_batch():
    """测试 batch 均值"""
    inputs = torch.tensor([
        [2.0, 1.0, 0.1],
        [0.1, 2.0, 1.0],
        [1.0, 0.1, 2.0],
    ])
    targets = torch.tensor([0, 1, 2])
    result = run_cross_entropy(inputs, targets)

    # 每个样本的 loss 应该相同（对称输入）
    single_loss = -torch.log(torch.softmax(inputs[0:1], dim=-1)[0, 0])
    assert torch.allclose(result, single_loss, atol=1e-6), "batch 均值不正确"
    print(f"✓ batch 均值测试通过 loss={result:.4f}")


def test_cross_entropy_vs_pytorch():
    """测试与 PyTorch F.cross_entropy 一致"""
    torch.manual_seed(42)
    inputs = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    expected = torch.nn.functional.cross_entropy(inputs, targets)
    result = run_cross_entropy(inputs, targets)

    assert torch.allclose(result, expected, atol=1e-6), f"与 PyTorch 不一致: {result} vs {expected}"
    print(f"✓ 与 PyTorch 一致性测试通过 loss={result:.4f}")


def test_cross_entropy_gradient():
    """测试梯度反向传播"""
    inputs = torch.randn(4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4,))

    loss = run_cross_entropy(inputs, targets)
    loss.backward()

    assert inputs.grad is not None, "梯度应存在"
    print("✓ 梯度反向传播正常")


if __name__ == "__main__":
    test_cross_entropy_basic()
    test_cross_entropy_numerical_stability()
    test_cross_entropy_formula()
    test_cross_entropy_batch()
    test_cross_entropy_vs_pytorch()
    test_cross_entropy_gradient()
    print("\n✅ Step 6 run_cross_entropy 实现正确！")
