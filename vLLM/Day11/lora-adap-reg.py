class LoRARegistry:
    SQL_KW   = ["select","insert","update","delete","join","where"]
    CODE_KW  = ["def ","class ","import ","function","algorithm"]
    LEGAL_KW = ["legal","contract","clause","liability","statute"]

    def __init__(self, base_model, base_port=8000):
        self.base_model = base_model
        self.base_port  = base_port
        self.adapters   = {}

    def register(self, name, path, rank, description=""):
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already registered.")
        self.adapters[name] = {
            "path": path, "rank": rank,
            "description": description, "request_count": 0
        }

    def unregister(self, name):
        if name not in self.adapters:
            raise KeyError(f"Adapter '{name}' not found.")
        del self.adapters[name]

    def route(self, prompt):
        pl = prompt.lower()
        if "sql" in self.adapters and any(kw in pl for kw in self.SQL_KW):
            return "sql"
        if "code" in self.adapters and any(kw in pl for kw in self.CODE_KW):
            return "code"
        if "legal" in self.adapters and any(kw in pl for kw in self.LEGAL_KW):
            return "legal"
        return self.base_model

    def get_model_name(self, prompt):
        name = self.route(prompt)
        if name in self.adapters:
            self.adapters[name]["request_count"] += 1
        return name

    def stats(self):
        print("=" * 58)
        print(f"  {'Adapter':<12} {'Rank':>5} {'Requests':>9}  Description")
        print("  " + "-" * 52)
        for name, info in self.adapters.items():
            print(f"  {name:<12} {info['rank']:>5} {info['request_count']:>9}  {info['description']}")
        print("=" * 58)


reg = LoRARegistry("meta-llama/Meta-Llama-3-8B-Instruct")
reg.register("sql",   "./adapters/sql",   rank=16, description="Text-to-SQL")
reg.register("code",  "./adapters/code",  rank=32, description="Code generation")
reg.register("legal", "./adapters/legal", rank=16, description="Legal drafting")

prompts = [
    "SELECT * FROM users WHERE age > 30",
    "Write a def function to sort a list",
    "Draft a liability clause for a contract",
    "What is the weather today?",
]
for p in prompts:
    model = reg.get_model_name(p)
    print(f"  [{model:10}] {p[:45]}")

print()
reg.stats()