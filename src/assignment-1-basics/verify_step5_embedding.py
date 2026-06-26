"""Step 5 验证: run_embedding 实现"""

import torch
import sys
sys.path.insert(0, '.')

from tests.adapters import run_embedding


def test_embedding_basic():
    """测试嵌入查找基本功能"""
    vocab_size, d_model = 10, 4
    weights = torch.randn(vocab_size, d_model)

    # 查找单个 token
    token_ids = torch.tensor([3])
    result = run_embedding(vocab_size, d_model, weights, token_ids)
    expected = weights[3]
    assert torch.allclose(result[0], expected), "单 token 查找不匹配"
    print("✓ 单 token 查找测试通过")


def test_embedding_batch():
    """测试批量查找"""
    vocab_size, d_model = 100, 16
    weights = torch.randn(vocab_size, d_model)
    token_ids = torch.tensor([42, 7, 13, 99, 0])

    result = run_embedding(vocab_size, d_model, weights, token_ids)
    assert result.shape == (5, d_model), f"形状错误: {result.shape}"

    # 每一行应对应 weights 中的对应行
    for i, tid in enumerate(token_ids):
        assert torch.allclose(result[i], weights[tid]), f"ID {tid} 查找不匹配"

    print("✓ 批量查找测试通过")


def test_embedding_2d_input():
    """测试 2D 输入 (batch, seq_len)"""
    vocab_size, d_model = 100, 16
    weights = torch.randn(vocab_size, d_model)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)

    result = run_embedding(vocab_size, d_model, weights, token_ids)
    assert result.shape == (2, 3, d_model), f"形状错误: {result.shape}"

    # 验证
    for i in range(2):
        for j in range(3):
            assert torch.allclose(result[i, j], weights[token_ids[i, j]])

    print("✓ 2D 输入测试通过")


def test_embedding_3d_input():
    """测试 3D 输入"""
    vocab_size, d_model = 50, 8
    weights = torch.randn(vocab_size, d_model)
    token_ids = torch.randint(0, vocab_size, (2, 3, 4))  # (2, 3, 4)

    result = run_embedding(vocab_size, d_model, weights, token_ids)
    assert result.shape == (2, 3, 4, d_model), f"形状错误: {result.shape}"
    print("✓ 3D 输入测试通过")


def test_embedding_zero_id():
    """测试 ID=0（边界情况）"""
    vocab_size, d_model = 10, 4
    weights = torch.randn(vocab_size, d_model)
    token_ids = torch.tensor([0])

    result = run_embedding(vocab_size, d_model, weights, token_ids)
    assert torch.allclose(result[0], weights[0]), "ID=0 查找失败"
    print("✓ ID=0 边界测试通过")


def test_embedding_last_id():
    """测试最后一个 ID（边界情况）"""
    vocab_size, d_model = 10, 4
    weights = torch.randn(vocab_size, d_model)
    token_ids = torch.tensor([9])  # vocab_size - 1

    result = run_embedding(vocab_size, d_model, weights, token_ids)
    assert torch.allclose(result[0], weights[9]), "最后一个 ID 查找失败"
    print("✓ 最后 ID 边界测试通过")


def test_embedding_vs_pytorch():
    """测试与 PyTorch nn.Embedding 一致"""
    vocab_size, d_model = 100, 16
    weights = torch.randn(vocab_size, d_model)

    layer = torch.nn.Embedding(vocab_size, d_model)
    layer.weight.data = weights.clone()

    token_ids = torch.tensor([10, 20, 30])
    expected = layer(token_ids)
    result = run_embedding(vocab_size, d_model, weights, token_ids)

    assert torch.allclose(result, expected), "与 PyTorch nn.Embedding 不一致"
    print("✓ 与 PyTorch 一致性测试通过")


if __name__ == "__main__":
    test_embedding_basic()
    test_embedding_batch()
    test_embedding_2d_input()
    test_embedding_3d_input()
    test_embedding_zero_id()
    test_embedding_last_id()
    test_embedding_vs_pytorch()
    print("\n✅ Step 5 run_embedding 实现正确！")
