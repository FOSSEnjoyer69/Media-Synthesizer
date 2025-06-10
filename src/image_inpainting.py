import os

import platform

from PIL import Image, ImageFilter

from image_editing import scale_image, combine_images, auto_resize_to_pil, overlay_image_using_metadata, crop_to_mask, paste_image
import gc
from icecream import ic

import random
from tqdm import tqdm

import cv2
import torch

from accelerate import cpu_offload
import numpy as np
from diffusers import (DDIMScheduler, KDPM2AncestralDiscreteScheduler, ControlNetModel, StableDiffusionControlNetInpaintPipeline)
from save import get_output_folder

from models import *

from lora_manager import STABLE_DIFFUSION_1_LORA_FOLDER_PATH

def run_image_inpaint(input_image:np.ndarray, input_mask:np.ndarray, input_pose:np.ndarray, 
                            prompt:str="", n_prompt:str="", 
                            sampling_steps:int=40, cfg_scale:float=4, seed: int = -1, iteration_count:int=2, 
                            inpaint_model_id: str = "Uminosachi/realisticVisionV51_v51VAE-inpainting", strength:float=1,
                            lora_model_paths:list=[], lora_strength:float=1,
                            min_inpaint_resolution:int=512, max_inpaint_resolution:int=1024):
    if input_mask is None:
        return [input_image] * iteration_count

    if input_pose is None:
        input_pose = np.zeros_like(input_image)

    save_folder_path:str = get_output_folder()

    if seed == -1:
        seed = random.randint(0, 2147483647)

    cv2.imwrite(f"{save_folder_path}/image.jpg", cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"{save_folder_path}/mask.jpg", cv2.cvtColor(input_mask, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"{save_folder_path}/pose.jpg", cv2.cvtColor(input_pose, cv2.COLOR_BGR2RGB))

    #Save Params To File
    params_file_path:str = f"{save_folder_path}/params.txt"
    with open(params_file_path, "w") as file:
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Negative Prompt: {n_prompt}\n")
        file.write(f"Sampling Steps: {sampling_steps}\n")
        file.write(f"CFG: {cfg_scale}\n")
        file.write(f"Seed: {seed}\n")
        file.write(f"Iteration Count: {iteration_count}\n")
        file.write(f"Inpaint Model: {inpaint_model_id}\n")

    generation_padding:float = 0.5
    image, crop_data = crop_to_mask(input_image, input_mask, generation_padding, min_resolution=min_inpaint_resolution)
    mask, _ = crop_to_mask(input_mask, input_mask, generation_padding, min_resolution=min_inpaint_resolution)
    pose, _ = crop_to_mask(input_pose, input_mask, generation_padding, min_resolution=min_inpaint_resolution)

    scaled_image = scale_image(image, max_inpaint_resolution, allow_upscale=False)
    mask_image = scale_image(mask, max_inpaint_resolution, allow_upscale=False)
    pose_image = scale_image(pose, max_inpaint_resolution, allow_upscale=False)

    # Save images for debugging
    cv2.imwrite(f"{save_folder_path}/scaled_input_image.jpg", cv2.cvtColor(np.array(scaled_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_folder_path}/scaled_mask_image.jpg", cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_folder_path}/scaled_pose_image.jpg", cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))

    if inpaint_model_id in STABLE_DIFFUSION_INPAINT_MODELS:
        generated_images:list = generate_inpaint_images(scaled_image, mask_image, pose_image, prompt, n_prompt, sampling_steps, cfg_scale, seed, iteration_count, inpaint_model_id, strength, lora_model_paths, lora_strength)
    else:
        return [input_image] * iteration_count


    result_iamges = []
    for i in range(len(generated_images)):
        result_image =  generated_images[i]
        result_image.save(f"{save_folder_path}/{i + 1}:generated_image.jpg")

        composite_image = combine_images(scaled_image, result_image, mask_image)
        composite_image.save(f"{save_folder_path}/{i + 1}:composite_image.jpg")

        output_image = overlay_image_using_metadata(input_image, composite_image, crop_data)
        output_image.save(f"{save_folder_path}/{i + 1}:output_image.jpg")
        result_iamges.append(output_image)

    return result_iamges

def generate_inpaint_images(image:Image.Image, mask_image:Image.Image, pose_image:Image.Image, 
                            prompt:str="", n_prompt:str="", 
                            sampling_steps:int=40, cfg_scale:float=4, seed: int = -1, iteration_count:int=2, 
                            inpaint_model_id: str = "Uminosachi/realisticVisionV51_v51VAE-inpainting", strength:float=1,
                            lora_model_paths:list=[], lora_strength:float=1):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not isinstance(mask_image, Image.Image):
        mask_image = Image.fromarray(mask_image)
    if not isinstance(pose_image, Image.Image):
        pose_image = Image.fromarray(pose_image)

    image, mask_image, pose_image = auto_resize_to_pil([image, mask_image, pose_image])


    generation_width, generation_height = (image.width, image.height)

    #openpose_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16, cache_dir="huggingface cache")
    openpose_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, cache_dir="huggingface cache")

    # Load the Stable Diffusion inpainting model with openpose_controlnet
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        inpaint_model_id, 
        controlnet=openpose_controlnet, 
        torch_dtype=torch.float16, 
        safety_checker=None,  
        requires_safety_checker=False,
        local_files_only=False,
        cache_dir="huggingface cache")

    for lora_model_path in lora_model_paths:
        pipe.load_lora_weights(f"{STABLE_DIFFUSION_1_LORA_FOLDER_PATH}/{lora_model_path}")

    # Use a specific scheduler for better quality
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    if seed == -1:
        seed = random.randint(0, 2147483647)

    output_images = []
    for i in range(iteration_count):
        image_seed = seed + i
        gc.collect()

        generator = torch.Generator(device="cuda").manual_seed(image_seed)
        # Generate the image
        output_image = pipe(
            prompt=[prompt],
            negative_prompt=[n_prompt],
            num_inference_steps=sampling_steps,
            generator=generator,
            cross_attention_kwargs={"scale": lora_strength},
            image=image,
            strength=strength,
            width=generation_width,
            height=generation_height,
            mask_image=mask_image,
            guidance_scale=cfg_scale,
            control_image=pose_image,
            
            
        ).images[0]
  
        output_images.append(output_image)

    gc.collect()

    return output_images
