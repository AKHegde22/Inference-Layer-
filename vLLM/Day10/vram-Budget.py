class VRAMBudgetPlanner:
    QUANT_MULTIPLIERS = {
        "fp16": 2.0, "fp8": 1.0,
        "awq": 0.6, "gptq": 0.6, "bitsandbytes": 0.65
    }

    def __init__(self, total_vram_gb):
        self.total_vram_gb = total_vram_gb

    def model_weights(self, params_b, quant_format):
        mult = self.QUANT_MULTIPLIERS.get(quant_format, 2.0)
        return params_b * mult

    def kv_cache(self, ctx_len, n_layers, n_heads, head_dim, dtype_bytes=2, n_seqs=1):
        total_bytes = 2 * n_layers * n_heads * head_dim * ctx_len * dtype_bytes * n_seqs
        return total_bytes / 1e9

    def cuda_overhead(self):
        return 1.5

    def plan(self, params_b, quant_format, ctx_len, n_layers, n_heads, head_dim, max_seqs=32):
        w  = self.model_weights(params_b, quant_format)
        kv = self.kv_cache(ctx_len, n_layers, n_heads, head_dim, n_seqs=max_seqs)
        oh = self.cuda_overhead()
        total = w + kv + oh
        return {
            "weights_gb":   round(w, 2),
            "kv_cache_gb":  round(kv, 2),
            "overhead_gb":  oh,
            "total_gb":     round(total, 2),
            "available_gb": self.total_vram_gb,
            "fits":         total <= self.total_vram_gb,
            "headroom_gb":  round(self.total_vram_gb - total, 2),
        }

    def print_plan(self, params_b, quant_format, ctx_len, n_layers, n_heads, head_dim, max_seqs=32):
        p = self.plan(params_b, quant_format, ctx_len, n_layers, n_heads, head_dim, max_seqs)
        status = "FITS" if p["fits"] else "DOES NOT FIT"
        print("=" * 48)
        print(f"  VRAM Budget Plan  [{status}]")
        print(f"  {params_b}B params | {quant_format.upper()} | ctx={ctx_len} | seqs={max_seqs}")
        print("=" * 48)
        print(f"  {'Model Weights':<22} {p['weights_gb']:>6.2f} GB")
        print(f"  {'KV Cache ({} seqs)'.format(max_seqs):<22} {p['kv_cache_gb']:>6.2f} GB")
        print(f"  {'CUDA Overhead':<22} {p['overhead_gb']:>6.2f} GB")
        print("  " + "-" * 30)
        print(f"  {'Total Required':<22} {p['total_gb']:>6.2f} GB")
        print(f"  {'Available':<22} {p['available_gb']:>6.2f} GB")
        print(f"  {'Headroom':<22} {p['headroom_gb']:>+6.2f} GB")
        print("=" * 48)


planner = VRAMBudgetPlanner(total_vram_gb=24)
planner.print_plan(
    params_b=7, quant_format="awq",
    ctx_len=8192, n_layers=32, n_heads=8, head_dim=128,
    max_seqs=32
)