import json, sys
from pathlib import Path
import ollama
sys.path.insert(0, "src")
from models import Script, engine_gate_errors

schema = json.loads(Path("config/schemas/script_schema.json").read_text())
resp = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "system", "content": Path("config/prompts/dsa_system_prompt.txt").read_text()},
        {"role": "user",   "content": json.dumps({"topic": "Stack", "facts": {}})},
    ],
    format=schema,                 # <-- grammar-constrained JSON, the whole point
    options={"temperature": 0.4},  # lower temp helps small models follow structure
)
raw = resp["message"]["content"]
script = Script.model_validate_json(raw)          # structural check
print("segments:", len(script.segments))
print("gate:", engine_gate_errors(script) or "PASS")