from pathlib import Path

STABLE_DIFFUSION_1_LORA_FOLDER_PATH = "Models/Lora/Stable Diffusion 1"

def get_stable_diffusion_1_lora_files():
    folder = Path(STABLE_DIFFUSION_1_LORA_FOLDER_PATH)
    files = [f.name for f in folder.iterdir() if f.is_file()]

    return files