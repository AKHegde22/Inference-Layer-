def get_sampling_params(use_case):
    params = {
        "code":     {"temp": 0.1, "top_k": 10, "top_p": 0.9,  "repeat_penalty": 1.0},
        "chat":     {"temp": 0.7, "top_k": 40, "top_p": 0.9,  "repeat_penalty": 1.1},
        "creative": {"temp": 0.9, "top_k": 0,  "top_p": 0.95, "repeat_penalty": 1.05},
        "factual":  {"temp": 0.0, "top_k": 1,  "top_p": 1.0,  "repeat_penalty": 1.0},
    }
    if use_case not in params:
        raise ValueError(f"Unknown use case: {use_case}. Choose from {list(params.keys())}")
    return params[use_case]

print(get_sampling_params("code"))
print(get_sampling_params("creative"))
print(get_sampling_params("chat"))
