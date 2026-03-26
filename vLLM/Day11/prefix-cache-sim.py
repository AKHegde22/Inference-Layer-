class PrefixCache:
    def __init__(self, block_size=16, max_blocks=50):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache      = {}

    def _tokenize(self, text):
        return [hash(w) % 1000 for w in text.split()]

    def _get_block_hashes(self, token_ids):
        hashes = []
        for i in range(0, len(token_ids), self.block_size):
            chunk = tuple(token_ids[i:i+self.block_size])
            hashes.append(hash(chunk) % 100000)
        return hashes

    def lookup(self, text):
        toks   = self._tokenize(text)
        hashes = self._get_block_hashes(toks)
        cached = sum(1 for h in hashes if h in self.cache)
        total  = len(hashes)
        return {
            "total_blocks":   total,
            "cached_blocks":  cached,
            "cache_hit_rate": cached / total if total else 0.0,
            "tokens_saved":   cached * self.block_size,
            "compute_needed": (total - cached) * self.block_size,
        }

    def store(self, text):
        toks   = self._tokenize(text)
        hashes = self._get_block_hashes(toks)
        for h in hashes:
            if h not in self.cache:
                if len(self.cache) >= self.max_blocks:
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                self.cache[h] = True

    def process(self, text):
        result = self.lookup(text)
        self.store(text)
        return result

    def batch_process(self, texts):
        return [self.process(t) for t in texts]


SYSTEM = "You are a helpful AI assistant with expertise in Python programming. " * 4

questions = [
    SYSTEM + " What is a list comprehension?",
    SYSTEM + " How do decorators work?",
    SYSTEM + " Explain generators.",
    SYSTEM + " What is a context manager?",
    SYSTEM + " How does asyncio work?",
]

cache = PrefixCache(block_size=16, max_blocks=100)
results = cache.batch_process(questions)

print(f"{'Request':<10} {'Hit Rate':>10} {'Tokens Saved':>14} {'Blocks Cached':>14}")
print("-" * 52)
for i, r in enumerate(results):
    print(f"  Q{i+1:<7} {r['cache_hit_rate']:>9.1%} {r['tokens_saved']:>14} {r['cached_blocks']:>14}/{r['total_blocks']}")