import json
def save_json(data: dict, file_path):
    with open(file_path, 'w') as json_file:
        json_obj = json.dumps(data)
        json_file.write(json_obj)


def load_json(file_path):
    with open(file_path, "r") as file:
        word2int = json.load(file)
    return word2int