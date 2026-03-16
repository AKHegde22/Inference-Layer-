def advise_config(vram_gb, ram_gb, cpu_cores, use_case="chat"):
    model_gb = 4.1
    ctx_map = {"chat": 4096, "rag": 8192, "agent": 16384}
    ctx_size = ctx_map.get(use_case, 4096)

    available = vram_gb - 1.0
    fraction = min(available / model_gb, 1.0)
    layers = int(fraction * 32)
    ngl = 99 if layers >= 32 else layers

    ctx_overhead_gb = ctx_size * 0.0005
    ram_ok = ram_gb >= (model_gb + ctx_overhead_gb)
    flash = ngl > 0

    print("=" * 50)
    print(f"  Performance Config Report ({use_case.upper()})")
    print("=" * 50)
    print(f"  GPU Layers (-ngl):    {ngl}")
    print(f"  CPU Threads (-t):     {cpu_cores}")
    print(f"  Context Size (-c):    {ctx_size}")
    print(f"  Flash Attention:      {'YES' if flash else 'NO'}")
    print(f"  RAM Sufficient:       {'YES' if ram_ok else 'WARNING: may swap!'}")
    print()
    cmd = (f"./build/bin/llama-server -m models/mistral.Q4_K_M.gguf "
           f"-ngl {ngl} -t {cpu_cores} -c {ctx_size}"
           + (" --flash-attn --mlock" if flash else "")
           + " --cont-batching --parallel 4 --host 0.0.0.0 --port 8080")
    print("  Recommended command:")
    print(f"  {cmd}")
    print("=" * 50)

advise_config(vram_gb=8, ram_gb=16, cpu_cores=8, use_case="rag")