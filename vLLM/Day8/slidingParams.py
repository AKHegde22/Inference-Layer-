def build_sampling_params(use_case, **overrides):
    presets = {
        "chat": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "max_tokens": 512, "repetition_penalty": 1.1,
        },
        "code": {
            "temperature": 0.1, "top_p": 0.95, "top_k": 10,
            "max_tokens": 1024, "repetition_penalty": 1.0, "stop": ["<|endoftext|>"],
        },
        "creative": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 0,
            "max_tokens": 800, "repetition_penalty": 1.05,
        },
        "factual": {
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_tokens": 256, "repetition_penalty": 1.0,
        },
        "batch_eval": {
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_tokens": 64, "repetition_penalty": 1.0, "n": 1, "seed": 42,
        },
    }
    if use_case not in presets:
        raise ValueError(f"Unknown use case '{use_case}'. Choose from: {list(presets)}")
    params = presets[use_case].copy()
    params.update(overrides)
    params["use_case"] = use_case
    return params

def describe_params(params_dict):
    descriptions = {
        "temperature":        "Controls randomness. Low=deterministic, high=creative.",
        "top_p":              "Nucleus sampling threshold — cumulative prob cutoff.",
        "top_k":              "Limit candidates to top K tokens (0=disabled).",
        "max_tokens":         "Maximum new tokens to generate.",
        "repetition_penalty": "Penalizes repeated tokens. >1.0 reduces repetitive loops.",
        "stop":               "Stop generation when any of these strings appear.",
        "seed":               "Fixed seed for reproducible deterministic outputs.",
        "n":                  "Number of output sequences per prompt.",
    }
    print(f"\nSamplingParams for use_case='{params_dict.get('use_case', '?')}':")
    print("─" * 50)
    for k, v in params_dict.items():
        if k == "use_case":
            continue
        desc = descriptions.get(k, "No description available.")
        print(f"  {k:22} = {str(v):10}  # {desc}")

p = build_sampling_params("code")
print(p)

p2 = build_sampling_params("chat", temperature=0.3, max_tokens=100)
print(p2)

describe_params(p)