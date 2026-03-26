import random

class SpecDecoder:
    def __init__(self, acceptance_rate=0.8, k=4, seed=42):
        self.acceptance_rate = acceptance_rate
        self.k   = k
        self.rng = random.Random(seed)

    def draft(self, position):
        return [f"tok_{position+i}" for i in range(self.k)]

    def verify(self, draft_tokens, acceptance_rate=None):
        rate = acceptance_rate if acceptance_rate is not None else self.acceptance_rate
        accepted = []
        for tok in draft_tokens:
            if self.rng.random() < rate:
                accepted.append(tok)
            else:
                break
        bonus = f"tok_bonus_{len(accepted)}"
        accepted.append(bonus)
        return accepted, len(accepted)

    def step(self, position):
        drafted = self.draft(position)
        tokens, n_acc = self.verify(drafted)
        return {
            "position":   position,
            "drafted":    self.k,
            "accepted":   n_acc,
            "tokens":     tokens,
            "efficiency": n_acc / self.k,
        }

    def simulate(self, total_tokens=50):
        position       = 0
        all_tokens     = []
        total_drafted  = 0
        total_accepted = 0
        steps          = 0

        while len(all_tokens) < total_tokens:
            result = self.step(position)
            steps          += 1
            total_drafted  += result["drafted"]
            total_accepted += result["accepted"]
            all_tokens.extend(result["tokens"])
            position += result["accepted"]

        all_tokens = all_tokens[:total_tokens]
        avg_tps = len(all_tokens) / steps
        return {
            "total_tokens":        total_tokens,
            "total_steps":         steps,
            "avg_tokens_per_step": round(avg_tps, 2),
            "effective_speedup":   round(avg_tps, 2),
            "total_drafted":       total_drafted,
            "total_accepted":      total_accepted,
            "acceptance_rate":     round(total_accepted / total_drafted, 3),
        }


for rate in [0.5, 0.7, 0.9]:
    dec = SpecDecoder(acceptance_rate=rate, k=4, seed=42)
    result = dec.simulate(total_tokens=100)
    print(f"Acceptance {rate:.0%}: "
          f"speedup={result['effective_speedup']:.2f}x  "
          f"steps={result['total_steps']}  "
          f"acc_rate={result['acceptance_rate']:.1%}")