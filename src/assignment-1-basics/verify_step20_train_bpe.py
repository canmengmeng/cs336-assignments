"""Step 20 验证: run_train_bpe 实现"""

import tempfile
import os
import sys
sys.path.insert(0, '.')

from tests.adapters import run_train_bpe, get_tokenizer


def test_train_bpe_basic():
    """测试基本训练功能"""
    # 写入测试语料
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("the cat sat on the mat\n")
        f.write("the cat ate the rat\n")
        corpus_path = f.name

    try:
        vocab, merges = run_train_bpe(corpus_path, vocab_size=300, special_tokens=['<|endoftext|>'])

        assert isinstance(vocab, dict), "vocab 应为字典"
        assert isinstance(merges, list), "merges 应为列表"
        assert len(vocab) <= 300, f"词表大小: {len(vocab)}"
        assert all(isinstance(m, tuple) and len(m) == 2 for m in merges), "merges 格式"
        print(f"✓ 基本训练测试通过 (词表大小: {len(vocab)}, 合并数: {len(merges)})")
    finally:
        os.unlink(corpus_path)


def test_train_bpe_vocab_structure():
    """测试词表结构"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello hello hello\n")
        corpus_path = f.name

    try:
        vocab, merges = run_train_bpe(corpus_path, vocab_size=260, special_tokens=['<|endoftext|>'])

        # 检查特殊 token
        special_bytes = '<|endoftext|>'.encode('utf-8')
        assert special_bytes in vocab.values(), "特殊 token 应在词表中"

        # 检查基础字节
        for i in range(256):
            assert bytes([i]) in vocab.values(), f"字节 {i} 应在词表中"

        print("✓ 词表结构测试通过")
    finally:
        os.unlink(corpus_path)


def test_train_bpe_merges():
    """测试合并规则"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # 重复的 pair 应该被合并
        f.write("ab ab ab ab ab\n")
        corpus_path = f.name

    try:
        vocab, merges = run_train_bpe(corpus_path, vocab_size=260, special_tokens=[])

        # 第一个合并应该是 (b'a', b'b')
        assert len(merges) > 0, "应该有合并"
        assert merges[0] == (b'a', b'b'), f"第一个合并: {merges[0]}"

        print(f"✓ 合并规则测试通过: {merges[:3]}...")
    finally:
        os.unlink(corpus_path)


def test_train_bpe_with_tokenizer():
    """测试训练后可以用于编码"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\n")
        corpus_path = f.name

    try:
        vocab, merges = run_train_bpe(corpus_path, vocab_size=300, special_tokens=['<|endoftext|>'])
        tokenizer = get_tokenizer(vocab, merges, ['<|endoftext|>'])

        # 编码-解码往返
        text = "hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text, f"往返失败: '{text}' → {ids} → '{decoded}'"
        print(f"✓ Tokenizer 集成测试通过: '{text}' → {ids} → '{decoded}'")
    finally:
        os.unlink(corpus_path)


def test_train_bpe_vocab_size():
    """测试词表大小限制"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("a b c d e f g h\n" * 100)
        corpus_path = f.name

    try:
        # 小词表
        vocab_small, merges_small = run_train_bpe(corpus_path, vocab_size=260, special_tokens=[])
        assert len(vocab_small) <= 260

        # 大词表
        vocab_large, merges_large = run_train_bpe(corpus_path, vocab_size=280, special_tokens=[])
        assert len(vocab_large) <= 280
        assert len(merges_large) >= len(merges_small)

        print(f"✓ 词表大小测试通过 (小: {len(vocab_small)}, 大: {len(vocab_large)})")
    finally:
        os.unlink(corpus_path)


def test_train_bpe_empty_corpus():
    """测试空语料"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        corpus_path = f.name

    try:
        vocab, merges = run_train_bpe(corpus_path, vocab_size=300, special_tokens=['<|endoftext|>'])
        assert len(vocab) == 257  # 256 bytes + 1 special
        assert len(merges) == 0
        print("✓ 空语料测试通过")
    finally:
        os.unlink(corpus_path)


if __name__ == "__main__":
    test_train_bpe_basic()
    test_train_bpe_vocab_structure()
    test_train_bpe_merges()
    test_train_bpe_with_tokenizer()
    test_train_bpe_vocab_size()
    test_train_bpe_empty_corpus()
    print("\n✅ Step 20 run_train_bpe 实现正确！")
    print("\n🎉 所有 20 个接口全部完成！")
