import sys, time, cProfile
sys.path.insert(0, '.')
from tests.adapters import run_train_bpe

# Profile the BPE training
cProfile.run('run_train_bpe("tests/fixtures/corpus.en", vocab_size=500, special_tokens=["endoftext"])', sort='cumulative')
