import json
from pathlib import Path

def open_json_file (file_path: str | Path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def save_json_file (file_path: str | Path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)