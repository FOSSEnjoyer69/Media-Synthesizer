from pathlib import Path

def get_lora_model_names():
    folder = Path('Models/Lora')
    files = [f.name for f in folder.iterdir() if f.is_file()]

    return files