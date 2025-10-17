import json

def open_json_file (file_path: str):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def save_json_file (file_path: str, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)