"""Step 16 验证: run_get_lr_cosine_schedule 实现"""

import math
import sys
sys.path.insert(0, '.')

from tests.adapters import run_get_lr_cosine_schedule


def test_lr_schedule_warmup():
    """测试预热阶段"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    # 预热开始: lr ≈ 0
    lr_0 = run_get_lr_cosine_schedule(0, max_lr, min_lr, warmup, cosine)
    assert abs(lr_0) < 1e-10, f"it=0: {lr_0}"

    # 预热中间: lr = max_lr * 0.5
    lr_50 = run_get_lr_cosine_schedule(50, max_lr, min_lr, warmup, cosine)
    assert abs(lr_50 - max_lr * 0.5) < 1e-10, f"it=50: {lr_50}"

    # 预热结束: lr = max_lr
    lr_100 = run_get_lr_cosine_schedule(100, max_lr, min_lr, warmup, cosine)
    assert abs(lr_100 - max_lr) < 1e-10, f"it=100: {lr_100}"

    print("✓ 预热阶段测试通过")


def test_lr_schedule_cosine():
    """测试余弦退火阶段"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    # 余弦开始: lr = max_lr
    lr_start = run_get_lr_cosine_schedule(100, max_lr, min_lr, warmup, cosine)
    assert abs(lr_start - max_lr) < 1e-10, f"cosine start: {lr_start}"

    # 余弦中间: lr = (max_lr + min_lr) / 2
    lr_mid = run_get_lr_cosine_schedule(550, max_lr, min_lr, warmup, cosine)
    expected_mid = (max_lr + min_lr) / 2
    assert abs(lr_mid - expected_mid) < 1e-10, f"cosine mid: {lr_mid}"

    # 余弦结束: lr = min_lr
    lr_end = run_get_lr_cosine_schedule(1000, max_lr, min_lr, warmup, cosine)
    assert abs(lr_end - min_lr) < 1e-10, f"cosine end: {lr_end}"

    print("✓ 余弦退火阶段测试通过")


def test_lr_schedule_plateau():
    """测试平稳阶段"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    lr_2000 = run_get_lr_cosine_schedule(2000, max_lr, min_lr, warmup, cosine)
    assert abs(lr_2000 - min_lr) < 1e-10, f"plateau: {lr_2000}"

    lr_10000 = run_get_lr_cosine_schedule(10000, max_lr, min_lr, warmup, cosine)
    assert abs(lr_10000 - min_lr) < 1e-10, f"plateau: {lr_10000}"

    print("✓ 平稳阶段测试通过")


def test_lr_schedule_monotonic_warmup():
    """测试预热阶段单调递增"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    prev_lr = 0
    for it in range(warmup):
        lr = run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup, cosine)
        assert lr >= prev_lr, f"预热应单调递增: it={it}, lr={lr}, prev={prev_lr}"
        prev_lr = lr

    print("✓ 预热单调性测试通过")


def test_lr_schedule_monotonic_cosine():
    """测试余弦阶段单调递减"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    prev_lr = max_lr + 1
    for it in range(warmup, cosine + 1):
        lr = run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup, cosine)
        assert lr <= prev_lr + 1e-10, f"余弦应单调递减: it={it}, lr={lr}, prev={prev_lr}"
        prev_lr = lr

    print("✓ 余弦单调性测试通过")


def test_lr_schedule_formula():
    """测试公式正确性"""
    max_lr, min_lr = 1e-3, 1e-5
    warmup, cosine = 100, 1000

    # 测试几个点
    test_cases = [0, 50, 100, 300, 550, 800, 1000, 2000]
    for it in test_cases:
        result = run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup, cosine)

        # 手动计算
        if it < warmup:
            expected = max_lr * (it / warmup)
        elif it <= cosine:
            t = (it - warmup) / (cosine - warmup)
            expected = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
        else:
            expected = min_lr

        assert abs(result - expected) < 1e-10, f"it={it}: {result} != {expected}"

    print("✓ 公式验证通过")


if __name__ == "__main__":
    test_lr_schedule_warmup()
    test_lr_schedule_cosine()
    test_lr_schedule_plateau()
    test_lr_schedule_monotonic_warmup()
    test_lr_schedule_monotonic_cosine()
    test_lr_schedule_formula()
    print("\n✅ Step 16 run_get_lr_cosine_schedule 实现正确！")
