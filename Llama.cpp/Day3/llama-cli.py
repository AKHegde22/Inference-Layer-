def build_cli_command(model_path, prompt, **kwargs):
    flag_map = {
        "temp": "--temp",
        "top_k": "--top-k",
        "top_p": "--top-p",
        "repeat_penalty": "--repeat-penalty",
        "max_tokens": "-n",
        "threads": "-t",
        "ctx_size": "-c",
    }
    cmd = f'./build/bin/llama-cli -m {model_path} -p "{prompt}"'
    for key, val in kwargs.items():
        if key in flag_map:
            cmd += f" {flag_map[key]} {val}"
    return cmd

cmd = build_cli_command(
    "models/mistral.gguf",
    "Write hello world in Python",
    temp=0.1, threads=8, max_tokens=200
)
print(cmd)
