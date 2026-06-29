import sys, time
sys.path.insert(0, '.')
from tests.adapters import run_train_bpe

start = time.time()
vocab, merges = run_train_bpe('tests/fixtures/corpus.en', vocab_size=500, special_tokens=['endoftext'])
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s, Vocab: {len(vocab)}, Merges: {len(merges)}')
