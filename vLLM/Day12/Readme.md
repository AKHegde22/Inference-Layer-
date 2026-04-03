Structured Output — Guaranteed JSON
WHY STRUCTURED OUTPUT MATTERS
══════════════════════════════════════════════════════

Without structure:
  Prompt: "Return user info as JSON"
  Response: "Sure! Here is the user info:
             {'name': 'Alice', age: 30}  ← invalid JSON (single quotes)
             Hope that helps!"           ← extra prose

With structured output:
  Response: {"name": "Alice", "age": 30}  ← always valid, always parseable

This is critical for:
  • LLM pipelines (parse output, feed to next step)
  • API responses (caller expects a schema)
  • Data extraction (entities, classifications)
  • Agent actions (function arguments)

THREE APPROACHES IN vLLM:

  1. response_format: json_object
     → Guarantees valid JSON, but any schema
     
  2. response_format: json_schema
     → Guarantees JSON matching YOUR exact schema
     → Uses guided decoding internally (XGrammar)
     
  3. guided_decoding (vLLM-native extra param)
     → Direct grammar/regex/schema control
     → More flexible than OpenAI spec

ENABLING GUIDED DECODING (server side):
  vllm serve model \
    --guided-decoding-backend xgrammar   # default, fast
    # or: --guided-decoding-backend outlines

  Both backends compile your schema/grammar into a token mask
  that forces the sampler to only pick valid next tokens.
  Zero chance of invalid output — it's mathematically impossible.

  JSON Schema Structured Output
USING JSON SCHEMA IN vLLM
══════════════════════════════════════════════════════

METHOD 1: response_format (OpenAI-compatible)

  from openai import OpenAI
  import json

  client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

  schema = {
      "type": "object",
      "properties": {
          "name":       {"type": "string"},
          "age":        {"type": "integer", "minimum": 0},
          "email":      {"type": "string", "format": "email"},
          "skills":     {"type": "array", "items": {"type": "string"}},
          "employed":   {"type": "boolean"}
      },
      "required": ["name", "age", "skills"]
  }

  response = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      messages=[{
          "role": "user",
          "content": "Extract: Alice is 30, a Python developer who loves ML"
      }],
      response_format={
          "type": "json_schema",
          "json_schema": {
              "name":   "person_schema",   # required name field
              "schema": schema,
              "strict": True
          }
      }
  )
  data = json.loads(response.choices[0].message.content)
  print(data["name"])    # "Alice"
  print(data["skills"])  # ["Python", "ML"]

METHOD 2: extra_body with guided_json (vLLM-native)

  response = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      messages=[{"role": "user", "content": "List 3 planets"}],
      extra_body={
          "guided_json": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 3,
              "maxItems": 3
          }
      }
  )

METHOD 3: Pydantic integration (cleanest Python API)

  from pydantic import BaseModel
  from typing import List, Optional

  class Person(BaseModel):
      name:     str
      age:      int
      skills:   List[str]
      email:    Optional[str] = None

  response = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      messages=[{"role":"user","content":"Describe Alice, 30, Python dev"}],
      extra_body={"guided_json": Person.model_json_schema()}
  )
  person = Person.model_validate_json(response.choices[0].message.content)
  print(person.name)    # Alice (typed Python object!)

  Regex, Grammar & Choice Constraints
BEYOND JSON — OTHER GUIDED DECODING MODES
══════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GUIDED REGEX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Force output to match a regex pattern.

  # Force a date in YYYY-MM-DD format
  response = client.chat.completions.create(
      model="local",
      messages=[{"role":"user","content":"What date is Christmas 2025?"}],
      extra_body={"guided_regex": r"\d{4}-\d{2}-\d{2}"}
  )
  # Output will ALWAYS be: 2025-12-25

  # Force a phone number
  extra_body={"guided_regex": r"\+1-\d{3}-\d{3}-\d{4}"}

  # Force a hex color
  extra_body={"guided_regex": r"#[0-9A-Fa-f]{6}"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. GUIDED CHOICE (classification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Restrict output to one of N exact strings.
  Perfect for classification tasks!

  response = client.chat.completions.create(
      model="local",
      messages=[{
          "role": "user",
          "content": "Classify sentiment: 'I love this product!'"
      }],
      extra_body={"guided_choice": ["positive", "negative", "neutral"]}
  )
  # Output: "positive"  (always exactly one of the choices)

  # Multi-class topic classification
  extra_body={"guided_choice": [
      "sports", "politics", "technology",
      "entertainment", "science", "business"
  ]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. GUIDED GRAMMAR (EBNF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Full grammar in EBNF (Extended Backus-Naur Form).
  More powerful than GBNF from llama.cpp.

  arithmetic_grammar = """
  start: expr
  expr:  term ("+" term | "-" term)*
  term:  NUMBER
  NUMBER: /[0-9]+/
  """

  response = client.chat.completions.create(
      model="local",
      messages=[{"role":"user","content":"Give me an arithmetic expression"}],
      extra_body={"guided_grammar": arithmetic_grammar}
  )
  # Output: "42+7-3"  (always valid arithmetic)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. WHITESPACE PATTERNS (json_object)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Simple valid JSON, any schema
  response_format={"type": "json_object"}

  # Control whitespace in output
  extra_body={"guided_whitespace_pattern": " "}  # compact

  Tool Calling & Function Calling
TOOL CALLING IN vLLM
══════════════════════════════════════════════════════

Tool calling lets the model request execution of functions.
The model outputs a structured call → you execute it → feed result back.

REQUIRES: models fine-tuned for tool use:
  • Llama 3.1 / 3.2 / 3.3  (Meta's official tool-call support)
  • Mistral Nemo / Large
  • Qwen 2.5 series
  • Hermes / Functionary (community fine-tunes)

BASIC TOOL CALLING:

  tools = [{
      "type": "function",
      "function": {
          "name": "get_weather",
          "description": "Get weather for a location",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string", "description": "City name"},
                  "unit":     {"type": "string", "enum": ["celsius","fahrenheit"]}
              },
              "required": ["location"]
          }
      }
  }]

  response = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3.1-8B-Instruct",
      messages=[{"role":"user","content":"What's the weather in Paris?"}],
      tools=tools,
      tool_choice="auto"   # "auto", "none", or {"type":"function","function":{"name":"..."}}
  )

  msg = response.choices[0].message
  if msg.tool_calls:
      call = msg.tool_calls[0]
      print(call.function.name)        # "get_weather"
      print(call.function.arguments)   # '{"location": "Paris", "unit": "celsius"}'

FULL AGENTIC LOOP:

  import json

  def execute_tool(name, args):
      if name == "get_weather":
          return {"temp": 18, "condition": "cloudy", "city": args["location"]}
      return {"error": "unknown tool"}

  history = [{"role":"user","content":"Weather in Tokyo and London?"}]

  while True:
      resp = client.chat.completions.create(
          model="meta-llama/Meta-Llama-3.1-8B-Instruct",
          messages=history, tools=tools, tool_choice="auto"
      )
      msg = resp.choices[0].message
      history.append(msg)

      if not msg.tool_calls:
          print("Final:", msg.content)
          break

      for call in msg.tool_calls:
          args   = json.loads(call.function.arguments)
          result = execute_tool(call.function.name, args)
          history.append({
              "role":         "tool",
              "tool_call_id": call.id,
              "content":      json.dumps(result)
          })

PARALLEL TOOL CALLS:
  # Model can request multiple tool calls at once!
  msg.tool_calls  # list — may contain [get_weather(Tokyo), get_weather(London)]
  # Execute them all in parallel, then return both results

  Vision-Language Models
MULTIMODAL INFERENCE IN vLLM
══════════════════════════════════════════════════════

vLLM supports vision-language models (VLMs) natively.
Images are encoded by a vision encoder → fed to the LLM.

SUPPORTED MODELS:
  • Llama 3.2 Vision (11B, 90B)  ← best quality
  • Qwen2-VL / Qwen2.5-VL        ← strong, efficient
  • LLaVA 1.5 / LLaVA-NeXT
  • InternVL2
  • Phi-3.5 Vision
  • SmolVLM (tiny, fast)

STARTING A VISION SERVER:
  vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85

SENDING IMAGES — BASE64:

  import base64, requests
  from openai import OpenAI

  client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

  def encode_image(path):
      with open(path, "rb") as f:
          return base64.b64encode(f.read()).decode("utf-8")

  response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-11B-Vision-Instruct",
      messages=[{
          "role": "user",
          "content": [
              {"type": "text",  "text": "What is in this image?"},
              {"type": "image_url",
               "image_url": {
                   "url": f"data:image/jpeg;base64,{encode_image('photo.jpg')}"
               }}
          ]
      }],
      max_tokens=300
  )
  print(response.choices[0].message.content)

SENDING IMAGES — URL:

  response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-11B-Vision-Instruct",
      messages=[{
          "role": "user",
          "content": [
              {"type": "text", "text": "Describe this chart in detail."},
              {"type": "image_url",
               "image_url": {"url": "https://example.com/chart.png"}}
          ]
      }]
  )

MULTI-IMAGE SUPPORT:
  # Some models support multiple images per message
  "content": [
      {"type": "text", "text": "Compare these two images:"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  ]

VISION + STRUCTURED OUTPUT:
  # Extract structured data FROM an image
  response = client.chat.completions.create(
      model="meta-llama/Llama-3.2-11B-Vision-Instruct",
      messages=[{
          "role": "user",
          "content": [
              {"type": "text",      "text": "Extract receipt data as JSON"},
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
          ]
      }],
      response_format={"type": "json_object"}
  )
