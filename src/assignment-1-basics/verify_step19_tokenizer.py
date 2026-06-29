"""Step 19 验证: get_tokenizer 实现"""

import sys
sys.path.insert(0, '.')

from tests.adapters import get_tokenizer


def make_simple_vocab():
    """构造简单词表"""
    vocab = {
        0: b'h',
        1: b'e',
        2: b'l',
        3: b'o',
        4: b'w',
        5: b'r',
        6: b'd',
        7: b' ',
        8: b'he',
        9: b'll',
        10: b'hell',    # he + ll = hell
        11: b'hello',   # hell + o = hello
        12: b'wo',
        13: b'wor',
        14: b'world',   # wor + ld = world
    }
    merges = [
        (b'h', b'e'),      # h + e → he
        (b'l', b'l'),      # l + l → ll
        (b'he', b'll'),    # he + ll → hell
        (b'hell', b'o'),   # hell + o → hello
        (b'w', b'o'),      # w + o → wo
        (b'wo', b'r'),     # wo + r → wor
        (b'wor', b'ld'),   # wor + ld → world
    ]
    return vocab, merges


def test_tokenizer_basic():
    """测试基本功能"""
    vocab, merges = make_simple_vocab()
    tokenizer = get_tokenizer(vocab, merges)

    # 编码
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list), "应返回列表"
    assert all(isinstance(i, int) for i in ids), "应为整数列表"

    # 解码
    text = tokenizer.decode(ids)
    assert isinstance(text, str), "应返回字符串"
    print("✓ 基本功能测试通过")


def test_tokenizer_roundtrip():
    """测试编码-解码往返"""
    vocab, merges = make_simple_vocab()
    tokenizer = get_tokenizer(vocab, merges)

    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text, f"往返失败: '{text}' → {ids} → '{decoded}'"
    print(f"✓ 往返测试通过: '{text}' → {ids} → '{decoded}'")


def test_tokenizer_encode():
    """测试编码过程"""
    vocab, merges = make_simple_vocab()
    tokenizer = get_tokenizer(vocab, merges)

    ids = tokenizer.encode("hello")
    # 应该被合并成单个 token 'hello' (id=11)
    assert ids == [11], f"hello → {ids}"
    print(f"✓ 编码测试通过: 'hello' → {ids}")


def test_tokenizer_decode():
    """测试解码过程"""
    vocab, merges = make_simple_vocab()
    tokenizer = get_tokenizer(vocab, merges)

    text = tokenizer.decode([11, 7, 14])  # hello + space + world
    assert text == "hello world", f"解码: '{text}'"
    print(f"✓ 解码测试通过: [11, 7, 14] → '{text}'")


def test_tokenizer_special_tokens():
    """测试特殊 token"""
    vocab = {
        0: b'hello',
        1: b'world',
        2: b'<|im_start|>',
    }
    merges = []
    special_tokens = ['<|im_start|>']

    tokenizer = get_tokenizer(vocab, merges, special_tokens)

    text = "<|im_start|>"
    ids = tokenizer.encode(text)
    assert ids == [2], f"特殊 token 编码: {ids}"
    print(f"✓ 特殊 token 测试通过: '{text}' → {ids}")


def test_tokenizer_unknown_chars():
    """测试未知字符处理"""
    vocab = {
        0: b'a',
        1: b'b',
    }
    merges = []

    tokenizer = get_tokenizer(vocab, merges)

    # 未知字符应该作为单字节处理
    ids = tokenizer.encode("ab")
    assert len(ids) == 2, f"ab → {ids}"
    print(f"✓ 未知字符测试通过: 'ab' → {ids}")


if __name__ == "__main__":
    test_tokenizer_basic()
    test_tokenizer_roundtrip()
    test_tokenizer_encode()
    test_tokenizer_decode()
    test_tokenizer_special_tokens()
    test_tokenizer_unknown_chars()
    print("\n✅ Step 19 get_tokenizer 实现正确！")
