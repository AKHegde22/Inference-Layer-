def calc_gpu_layers(model_size_gb, vram_gb, total_layers=32, overhead_gb=1.0):
    available = vram_gb - overhead_gb
    fraction = available / model_size_gb
    layers = int(fraction * total_layers)
    layers = min(layers, total_layers)
    return 99 if layers >= total_layers else layers

def generate_server_cmd(model_path, vram_gb, model_size_gb, n_cores, ctx_size=8192, port=8080):
    ngl = calc_gpu_layers(model_size_gb, vram_gb)
    parts = [
        "./build/bin/llama-server",
        f"-m {model_path}",
        f"-ngl {ngl}",
        f"-t {n_cores}",
        f"-c {ctx_size}",
        "--parallel 4",
        "--cont-batching",
        f"--host 0.0.0.0",
        f"--port {port}",
    ]
    if ngl > 0:
        parts.insert(3, "--flash-attn")
        parts.insert(4, "--mlock")
    return " \\".join(parts)

print(generate_server_cmd("models/llama3.gguf", 8.0, 4.9, 8))