def kv_cache_mb(ctx_len, n_layers, n_heads, head_dim, dtype_bytes=2):
    total_bytes = 2 * n_layers * n_heads * head_dim * ctx_len * dtype_bytes
    return total_bytes / (1024 * 1024)

print(f"4K ctx:  {kv_cache_mb(4096,  32, 8, 128):.0f} MB")
print(f"32K ctx: {kv_cache_mb(32768, 32, 8, 128):.0f} MB")