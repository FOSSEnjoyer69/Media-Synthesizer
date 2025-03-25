from datetime import datetime
import os

output_folder:str = "output"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def get_output_folder() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    save_folder_path = f"{output_folder}/{formatted_time}"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    return save_folder_path
