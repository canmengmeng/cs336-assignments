"""Step 17-18 验证: run_save_checkpoint & run_load_checkpoint 实现"""

import torch
import tempfile
import sys
sys.path.insert(0, '.')

from tests.adapters import run_save_checkpoint, run_load_checkpoint, get_adamw_cls


def make_simple_model():
    """构造简单模型"""
    model = torch.nn.Linear(4, 2)
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def test_save_checkpoint_basic():
    """测试基本保存功能"""
    model, optimizer = make_simple_model()
    iteration = 42

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        run_save_checkpoint(model, optimizer, iteration, f.name)

        # 验证文件存在且可加载
        checkpoint = torch.load(f.name, weights_only=True)
        assert 'model' in checkpoint, "应包含 model"
        assert 'optimizer' in checkpoint, "应包含 optimizer"
        assert 'iteration' in checkpoint, "应包含 iteration"
        assert checkpoint['iteration'] == 42, f"iteration: {checkpoint['iteration']}"
    print("✓ 基本保存测试通过")


def test_save_checkpoint_model_state():
    """测试模型状态正确保存"""
    model, optimizer = make_simple_model()

    # 修改模型参数
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(2.0)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        run_save_checkpoint(model, optimizer, 0, f.name)

        checkpoint = torch.load(f.name, weights_only=True)
        saved_state = checkpoint['model']

        assert torch.allclose(saved_state['weight'], torch.ones_like(model.weight))
        assert torch.allclose(saved_state['bias'], torch.ones_like(model.bias) * 2)
    print("✓ 模型状态保存测试通过")


def test_save_checkpoint_file_like():
    """测试文件对象保存"""
    model, optimizer = make_simple_model()

    import io
    buffer = io.BytesIO()
    run_save_checkpoint(model, optimizer, 100, buffer)

    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    assert checkpoint['iteration'] == 100
    print("✓ 文件对象保存测试通过")


def test_load_checkpoint_basic():
    """测试基本加载功能"""
    model, optimizer = make_simple_model()

    # 修改参数并保存
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(4.0)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        run_save_checkpoint(model, optimizer, 99, f.name)

        # 创建新模型并加载
        model2, optimizer2 = make_simple_model()
        iteration = run_load_checkpoint(model2, optimizer2, f.name)

        assert iteration == 99, f"iteration: {iteration}"
        assert torch.allclose(model2.weight, model.weight), "权重不匹配"
        assert torch.allclose(model2.bias, model.bias), "偏置不匹配"
    print("✓ 基本加载测试通过")


def test_load_checkpoint_roundtrip():
    """测试保存-加载往返"""
    model, optimizer = make_simple_model()

    # 训练几步
    x = torch.randn(2, 4)
    for _ in range(5):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 保存
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        run_save_checkpoint(model, optimizer, 5, f.name)

        # 加载到新模型
        model2, optimizer2 = make_simple_model()
        iter2 = run_load_checkpoint(model2, optimizer2, f.name)

        assert iter2 == 5
        assert torch.allclose(model2.weight, model.weight)
        assert torch.allclose(model2.bias, model.bias)
    print("✓ 往返测试通过")


if __name__ == "__main__":
    test_save_checkpoint_basic()
    test_save_checkpoint_model_state()
    test_save_checkpoint_file_like()
    test_load_checkpoint_basic()
    test_load_checkpoint_roundtrip()
    print("\n✅ Step 17-18 run_save_checkpoint & run_load_checkpoint 实现正确！")
