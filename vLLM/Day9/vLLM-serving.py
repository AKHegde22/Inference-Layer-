def build_vllm_command(model, **kwargs):
    FLAG_MAP = {
        "host":                "--host",
        "port":                "--port",
        "dtype":               "--dtype",
        "max_model_len":       "--max-model-len",
        "gpu_memory_util":     "--gpu-memory-utilization",
        "tensor_parallel":     "--tensor-parallel-size",
        "max_num_seqs":        "--max-num-seqs",
        "served_name":         "--served-model-name",
        "quantization":        "--quantization",
        "api_key":             "--api-key",
        "enable_prefix_cache": "--enable-prefix-caching",
        "disable_log":         "--disable-log-requests",
    }
    BOOL_FLAGS = {"enable_prefix_cache", "disable_log"}

    parts = [f"vllm serve {model}"]
    for key, val in kwargs.items():
        if key not in FLAG_MAP:
            continue
        flag = FLAG_MAP[key]
        if key in BOOL_FLAGS:
            if val:
                parts.append(flag)
        else:
            parts.append(f"{flag} {val}")

    return " \\\n  ".join(parts)


print(build_vllm_command(
    "mistralai/Mistral-7B-Instruct-v0.2",
    host="0.0.0.0", port=8000, max_model_len=4096,
    gpu_memory_util=0.90, served_name="mistral-7b",
    enable_prefix_cache=True, disable_log=True
))