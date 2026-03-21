"""
Day 9 — vLLM server health check and model info
Mirrors Day 4's api-health.py but for the vLLM server.
"""

import requests

BASE_URL = "http://localhost:8000"


def check_health():
    """GET /health — returns 200 when the server is ready."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        status = "UNREACHABLE"
    print(f"  Health:  {status}")
    return status == "OK"


def list_models():
    """GET /v1/models — lists loaded models and their metadata."""
    r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    r.raise_for_status()
    models = r.json()["data"]
    print(f"\n  Loaded models ({len(models)}):")
    for m in models:
        print(f"    id         : {m['id']}")
        print(f"    object     : {m['object']}")
        print(f"    created    : {m['created']}")
        print()
    return models


def server_info():
    """
    GET /v1/models also returns vLLM-specific fields like max_model_len.
    This function extracts and prints them cleanly.
    """
    r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    r.raise_for_status()
    for m in r.json()["data"]:
        print(f"  Model ID        : {m['id']}")
        # vLLM adds these extra fields beyond the OpenAI spec
        print(f"  Max model len   : {m.get('max_model_len', 'N/A')}")
        print(f"  Quantization    : {m.get('quantization', 'none')}")


if __name__ == "__main__":
    print("=" * 40)
    print("  vLLM Server Health Check")
    print("=" * 40)

    alive = check_health()
    if not alive:
        print("\n  Server not running. Start it with:")
        print("  python -m vllm.entrypoints.openai.api_server --model <model_id>")
    else:
        list_models()
        server_info()
