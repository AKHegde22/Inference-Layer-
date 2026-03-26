def validate_requirements(req):
    warnings = []
    if req.get("use_spec_decode") and req.get("use_chunked_prefill"):
        warnings.append("Speculative decoding + chunked prefill may conflict — test before production.")
    if req.get("gpu_util", 0.85) > 0.95:
        warnings.append(f"gpu_util={req['gpu_util']} is very high — risk of OOM on traffic spikes.")
    if req.get("lora_adapters") and not req.get("use_lora"):
        warnings.append("lora_adapters specified but use_lora is False — adapters will be ignored.")
    return warnings

def generate_advanced_config(req):
    model = req.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
    port  = req.get("port", 8000)
    cmd   = [f"vllm serve {model}"]
    pyc   = {"model": model}

    if q := req.get("quantization"):
        cmd.append(f"--quantization {q}")
        pyc["quantization"] = q

    gpu_util = req.get("gpu_util", 0.85)
    cmd.append(f"--gpu-memory-utilization {gpu_util}")
    pyc["gpu_memory_utilization"] = gpu_util

    if v := req.get("max_model_len"):
        cmd.append(f"--max-model-len {v}")
        pyc["max_model_len"] = v

    if v := req.get("max_num_seqs"):
        cmd.append(f"--max-num-seqs {v}")

    if req.get("use_lora"):
        cmd.append("--enable-lora")
        pyc["enable_lora"] = True
        rank = req.get("max_lora_rank", 16)
        cmd.append(f"--max-lora-rank {rank}")
        pyc["max_lora_rank"] = rank
        if adapters := req.get("lora_adapters"):
            mods = " ".join(f"{k}={v}" for k, v in adapters.items())
            cmd.append(f"--lora-modules {mods}")

    if req.get("use_prefix_cache"):
        cmd.append("--enable-prefix-caching")
        pyc["enable_prefix_caching"] = True

    if req.get("use_chunked_prefill"):
        cmd.append("--enable-chunked-prefill")

    if req.get("use_spec_decode"):
        spec = req.get("spec_model", "[ngram]")
        toks = req.get("spec_tokens", 4)
        cmd.append(f"--speculative-model {spec}")
        cmd.append(f"--num-speculative-tokens {toks}")
        pyc["speculative_model"] = spec
        pyc["num_speculative_tokens"] = toks

    cmd.append(f"--host 0.0.0.0 --port {port}")
    return {"command": " \\\n  ".join(cmd), "python_config": pyc}


req = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "quantization": "awq", "use_lora": True,
    "lora_adapters": {"sql": "./adapters/sql", "code": "./adapters/code"},
    "max_lora_rank": 32, "use_prefix_cache": True,
    "use_spec_decode": True, "spec_model": "[ngram]",
    "spec_tokens": 5, "gpu_util": 0.85,
    "max_model_len": 8192, "max_num_seqs": 128, "port": 8000,
}

warnings = validate_requirements(req)
if warnings:
    for w in warnings: print(f"  WARN: {w}")

result = generate_advanced_config(req)
print("\n--- CLI Command ---")
print(result["command"])
print("\n--- Python Config ---")
for k, v in result["python_config"].items():
    print(f"  {k} = {repr(v)}")