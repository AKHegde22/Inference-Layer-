import json
import math
from openai import OpenAI

def search_web(query):
    words = query.lower().split()
    return {"results": [
        {"title": f"Result about {words[0]}", "snippet": f"Information on {query}..."},
        {"title": f"More on {words[-1]}", "snippet": f"Details about {query} from experts..."},
    ]}

def get_stock_price(symbol):
    price  = abs(hash(symbol)) % 500 + 100
    change = round((abs(hash(symbol+symbol)) % 20 - 10) / 10, 2)
    return {"symbol": symbol.upper(), "price": round(price, 2), "change_pct": change}

def calculate(expression):
    try:
        safe_globals = {"__builtins__": {}}
        safe_locals  = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_locals.update({"abs": abs, "round": round})
        result = eval(expression, safe_globals, safe_locals)
        return {"result": round(float(result), 4), "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}

TOOLS = [
    {"type":"function","function":{
        "name":"search_web","description":"Search the web for information",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{
        "name":"get_stock_price","description":"Get current stock price for a ticker symbol",
        "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
    {"type":"function","function":{
        "name":"calculate","description":"Evaluate a mathematical expression",
        "parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression like '150 * 0.15'"}},"required":["expression"]}}},
]

class Agent:
    def __init__(self, model="local", port=8000, max_steps=5):
        self.client    = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
        self.model     = model
        self.max_steps = max_steps

    def execute_tool(self, name, args_str):
        args = json.loads(args_str)
        fn_map = {"search_web": search_web, "get_stock_price": get_stock_price, "calculate": calculate}
        result = fn_map.get(name, lambda **kw: {"error": f"Unknown: {name}"}**args) if name in fn_map else {"error": f"Unknown: {name}"}
        if name in fn_map:
            result = fn_map[name](**args)
        return json.dumps(result)

    def run(self, user_message):
        history = [{"role": "user", "content": user_message}]
        for step in range(self.max_steps):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=500,
            )
            msg = resp.choices[0].message
            history.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in (msg.tool_calls or [])]})
            if not msg.tool_calls:
                return msg.content
            for call in msg.tool_calls:
                result = self.execute_tool(call.function.name, call.function.arguments)
                print(f"  [tool] {call.function.name}({call.function.arguments[:40]}) → {result[:60]}")
                history.append({"role":"tool","tool_call_id":call.id,"content":result})
        return "Max steps reached."

agent = Agent()
result = agent.run("What is the stock price of NVDA, and what is 15% of that price?")
print("\nFinal:", result)