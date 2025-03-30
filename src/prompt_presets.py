import os

prompt_presets_folder_path = "settings/prompts/presets"
image_inpaiting_prompts_preset_folder = f"{prompt_presets_folder_path}/image inpainting"

def get_image_inpaint_preset_file_names():
    files = [f for f in os.listdir(image_inpaiting_prompts_preset_folder) if f.endswith('.txt')]

    return files

def get_image_inpaint_preset_file_paths():
    files = [os.path.join(image_inpaiting_prompts_preset_folder, f) for f in os.listdir(image_inpaiting_prompts_preset_folder) if f.endswith('.txt')]

    return files

def load_prompts(prompt_file_name:str) -> (str, str):
    path = f"{image_inpaiting_prompts_preset_folder}/{prompt_file_name}"

    prompts = []

    with open(path, 'r') as file:
        for line in file:
            prompts.append(line)

    return prompts[0], prompts[1]