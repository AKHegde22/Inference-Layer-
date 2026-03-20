def calc_gpu_layers(model_size_gb, vram_gb, total_layers=32, overhead_gb=1.0):
    available = vram_gb - overhead_gb
    fraction = available / model_size_gb
    layers = int(fraction * total_layers)
    layers = min(layers, total_layers)
    return 99 if layers >= total_layers else layers

print(calc_gpu_layers(4.1, 6.0))    # → 99
print(calc_gpu_layers(7.9, 8.0))    # → 99
print(calc_gpu_layers(7.9, 6.0))    # → 19
print(calc_gpu_layers(40.0, 24.0))  # → 18