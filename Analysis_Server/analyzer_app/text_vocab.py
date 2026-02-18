import json
from pathlib import Path

path = Path(__file__).parent / "vocab.json"
with open(path, "r") as f:
    word2idx = json.load(f)

