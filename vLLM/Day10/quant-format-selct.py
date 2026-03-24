def select_quantization(gpu_vram_gb, model_params_b, gpu_arch, priority="quality"):
    vram = {
        "fp16":         model_params_b * 2.0,
        "fp8":          model_params_b * 1.0,
        "awq":          model_params_b * 0.6,
        "gptq":         model_params_b * 0.6,
        "bitsandbytes": model_params_b * 0.65,
    }

    def fits(fmt):
        return vram[fmt] <= gpu_vram_gb

    if gpu_arch == "h100" and fits("fp8") and priority in ("speed", "quality"):
        return {"format": "fp8", "vram_needed": vram["fp8"],
                "reason": "H100 native FP8 — best quality and fastest on this GPU."}

    if fits("fp16") and priority != "memory":
        return {"format": "fp16", "vram_needed": vram["fp16"],
                "reason": "Full precision FP16 fits — no quality loss."}

    if fits("awq"):
        return {"format": "awq", "vram_needed": vram["awq"],
                "reason": "AWQ INT4 — best quality INT4, fast GEMM kernels."}

    if fits("gptq"):
        return {"format": "gptq", "vram_needed": vram["gptq"],
                "reason": "GPTQ INT4 — good quality, widely compatible."}

    if fits("bitsandbytes"):
        return {"format": "bitsandbytes", "vram_needed": vram["bitsandbytes"],
                "reason": "BitsAndBytes NF4 — no pre-quantization needed, slower inference."}

    return {"format": "does_not_fit", "vram_needed": vram["bitsandbytes"],
            "reason": f"Model needs ~{vram['bitsandbytes']:.1f}GB minimum but only {gpu_vram_gb}GB available."}


print(select_quantization(80, 70, "h100",   "speed"))
print(select_quantization(24, 7,  "rtx4090","quality"))
print(select_quantization(8,  7,  "a10g",   "memory"))
print(select_quantization(6,  13, "rtx3090","quality"))
print(select_quantization(4,  70, "other",  "memory"))