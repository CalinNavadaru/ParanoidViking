import json
from pathlib import Path

path = Path(__file__).parent / "char_vocab.json"
with open(path, "r") as f:
    char2idx = json.load(f)

PAD = 0