import math

class ContinuousBatchScheduler:
    def __init__(self, max_running=4, max_blocks=20, block_size=16):
        self.max_running = max_running
        self.max_blocks  = max_blocks
        self.block_size  = block_size
        self.waiting     = []
        self.running     = []
        self.finished    = []

    def add_request(self, req_id, prompt_len, max_new_tokens):
        blocks = math.ceil(prompt_len / self.block_size)
        self.waiting.append({
            "id": req_id, "prompt_len": prompt_len,
            "max_new": max_new_tokens, "generated": 0, "blocks_used": blocks
        })

    def step(self):
        # 1. Finish completed requests and free blocks
        still_running, finished_ids = [], []
        for req in self.running:
            if req["generated"] >= req["max_new"]:
                finished_ids.append(req["id"])
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. Promote waiting → running while budget allows
        scheduled_ids = []
        while (self.waiting and
               len(self.running) < self.max_running and
               self.total_blocks_used + self.waiting[0]["blocks_used"] <= self.max_blocks):
            req = self.waiting.pop(0)
            self.running.append(req)
            scheduled_ids.append(req["id"])

        # 3. Increment token generation for all running
        for req in self.running:
            req["generated"] += 1

        return {"scheduled": scheduled_ids, "finished": finished_ids, "waiting": len(self.waiting)}

    def status(self):
        print(f"  Running ({len(self.running)}): {[r['id'] for r in self.running]}")
        print(f"  Waiting ({len(self.waiting)}): {[r['id'] for r in self.waiting]}")
        print(f"  Blocks used: {self.total_blocks_used}/{self.max_blocks}")

    @property
    def total_blocks_used(self):
        return sum(r["blocks_used"] for r in self.running)


sched = ContinuousBatchScheduler(max_running=3, max_blocks=15, block_size=16)
for i, (pl, mt) in enumerate([(32,5),(48,3),(16,8),(64,4),(32,6),(16,2)], 1):
    sched.add_request(f"R{i}", pl, mt)

for step in range(6):
    result = sched.step()
    print(f"\nStep {step+1}: scheduled={result['scheduled']} finished={result['finished']}")
    sched.status()