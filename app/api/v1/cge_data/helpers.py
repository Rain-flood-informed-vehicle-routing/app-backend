import os
import json

def load_specific_json(year, month, day, root_directory):
    
    file_name = f"{year}_{month:02}_{day:02}.json"
    file_path = os.path.join(root_directory, str(year), f"{month:02}", file_name)
    print(file_path)

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    else:
        print(f"O arquivo {file_name} não foi encontrado.")
        return None