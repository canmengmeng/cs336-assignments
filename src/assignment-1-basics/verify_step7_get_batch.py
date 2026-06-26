"""Step 7 验证: run_get_batch 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_get_batch


def test_get_batch_shapes():
    """测试输出形状"""
    dataset = torch.arange(1000)
    batch_size, ctx_len = 8, 16

    inputs, labels = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))

    assert inputs.shape == (batch_size, ctx_len), f"inputs 形状: {inputs.shape}"
    assert labels.shape == (batch_size, ctx_len), f"labels 形状: {labels.shape}"
    print("✓ 形状测试通过")


def test_get_batch_labels_offset():
    """测试 labels = inputs 向右平移一位"""
    dataset = torch.arange(1000)
    batch_size, ctx_len = 4, 8

    inputs, labels = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))

    # 对于每个样本，labels[i] 应该等于 inputs[i] 向右偏移 1
    # 即 labels[i, j] = inputs[i, j+1] (除了最后一个位置无法验证)
    for i in range(batch_size):
        for j in range(ctx_len - 1):
            # inputs[i, j+1] 应该等于 labels[i, j]
            # 因为 inputs 是 [start...start+ctx-1]
            # labels 是 [start+1...start+ctx]
            pass  # 无法直接比较，因为 inputs 和 labels 来自不同位置

    # 换个思路: 验证 labels 的每个元素比 inputs 对应元素大 1
    # (因为 dataset 是 0,1,2,3,...)
    diff = labels - inputs
    # diff 应该全部为 1（因为连续序列，labels 右移一位）
    assert (diff == 1).all(), f"labels 应比 inputs 大 1: {diff}"
    print("✓ 标签偏移测试通过")


def test_get_batch_randomness():
    """测试随机性（不同调用应产生不同批次）"""
    dataset = torch.arange(10000)
    batch_size, ctx_len = 4, 8

    inputs1, _ = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))
    inputs2, _ = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))

    # 两次调用的起始位置大概率不同
    assert not torch.equal(inputs1, inputs2), "两次采样应不同"
    print("✓ 随机性测试通过")


def test_get_batch_boundary():
    """测试边界情况"""
    # 最小 dataset
    dataset = torch.arange(10)
    batch_size, ctx_len = 1, 8

    inputs, labels = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))
    assert inputs.shape == (1, 8)
    assert labels.shape == (1, 8)
    print("✓ 边界测试通过")


def test_get_batch_device():
    """测试 device 参数"""
    dataset = torch.arange(100)
    batch_size, ctx_len = 2, 4

    inputs, labels = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))
    assert inputs.device == torch.device('cpu')
    assert labels.device == torch.device('cpu')
    print("✓ device 测试通过")


def test_get_batch_values_in_range():
    """测试采样值在 dataset 范围内"""
    dataset = torch.arange(100)
    batch_size, ctx_len = 8, 16

    inputs, labels = run_get_batch(dataset, batch_size, ctx_len, device=torch.device('cpu'))

    assert inputs.min() >= 0, "inputs 有负值"
    assert inputs.max() < 100, "inputs 超出范围"
    assert labels.min() >= 0, "labels 有负值"
    assert labels.max() < 100, "labels 超出范围"
    print("✓ 值范围测试通过")


if __name__ == "__main__":
    test_get_batch_shapes()
    test_get_batch_labels_offset()
    test_get_batch_randomness()
    test_get_batch_boundary()
    test_get_batch_device()
    test_get_batch_values_in_range()
    print("\n✅ Step 7 run_get_batch 实现正确！")
