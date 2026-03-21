import time
import requests
from openai import OpenAI

class VLLMClient:
    def __init__(self, host="localhost", port=8000, api_key="none"):
        self.base_url = f"http://{host}:{port}"
        self.client   = OpenAI(base_url=f"{self.base_url}/v1", api_key=api_key)

    def health_check(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            if r.status_code == 200:
                print(f"Server OK at {self.base_url}")
                return True
            return False
        except Exception as e:
            print(f"Server unreachable: {e}")
            return False

    def list_models(self):
        try:
            response = self.client.models.list()
            models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by}
                      for m in response.data]
            print(f"{'Model ID':<30} {'Owned By':<15}")
            print("-" * 47)
            for m in models:
                print(f"{m['id']:<30} {m['owned_by']:<15}")
            return models
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []

    def get_model_info(self, model_id):
        models = self.list_models()
        for m in models:
            if m["id"] == model_id:
                return m
        return None

    def wait_for_ready(self, timeout=60, interval=2):
        start = time.time()
        print("Waiting for server", end="", flush=True)
        while time.time() - start < timeout:
            if self.health_check():
                print(" Ready!")
                return True
            print(".", end="", flush=True)
            time.sleep(interval)
        print(" Timed out.")
        return False


c = VLLMClient()
is_up = c.health_check()
print("Server up:", is_up)
if is_up:
    models = c.list_models()
    info = c.get_model_info("mistral-7b")
    print("Found:", info)