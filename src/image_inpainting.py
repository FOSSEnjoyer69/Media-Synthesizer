import os

import platform

from PIL import Image, ImageFilter

from image_editing import scale_image, combine_images, auto_resize_to_pil, create_borderd_image, crop_to_mask, find_image_in_large_image, paste_image, create_black_copy_of_image
import gc
from icecream import ic

import random
from tqdm import tqdm

import cv2
import torch

import numpy as np
from diffusers import  (KDPM2AncestralDiscreteScheduler,
                        StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline)

from save import get_output_folder

def generate_inpaint_images(input_image:np.ndarray, input_mask, input_pose, 
                            prompt:str="", n_prompt:str="", 
                            sampling_steps:int=40, cfg_scale:float=4, seed: int = -1, iteration_count:int=2, 
                            inpaint_model_id: str = "Uminosachi/realisticVisionV51_v51VAE-inpainting",
                            lora_model_paths:list=[], lora_strength:float=1,
                            min_inpaint_resolution:int=512, max_inpaint_resolution:int=1024):
    if input_mask is None:
        return [input_image] * iteration_count

    if input_pose is None:
        input_pose = np.zeros_like(input_image)

    save_folder_path:str = get_output_folder()
    inpaint_resolution:int = 1000

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
        file.write(f"Inpaint Resolution: {inpaint_resolution}\n")
    
    generation_padding:float = 0.5
    image = crop_to_mask(input_image, input_mask, generation_padding, min_resolution=min_inpaint_resolution)
    mask = crop_to_mask(input_mask, input_mask, generation_padding, min_resolution=min_inpaint_resolution)
    pose = crop_to_mask(input_pose, input_mask, generation_padding, min_resolution=min_inpaint_resolution)

    top_left, bottom_right = find_image_in_large_image(input_image, image)

    scaled_image = scale_image(image, max_inpaint_resolution, allow_upscale=False)
    mask_image = scale_image(mask, max_inpaint_resolution, allow_upscale=False)
    pose_image = scale_image(pose, max_inpaint_resolution, allow_upscale=False)
    
    scaled_image, mask_image, pose_image = auto_resize_to_pil([scaled_image, mask_image, pose_image])
    generation_width, generation_height = scaled_image.size


    # Save images for debugging
    scaled_image.save(f"{save_folder_path}/scaled_input_image.jpg")
    mask_image.save(f"{save_folder_path}/scaled_mask_image.jpg")
    pose_image.save(f"{save_folder_path}/scaled_pose_image.jpg")

    #openpose_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16, cache_dir="huggingface cache")
    openpose_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, cache_dir="huggingface cache")

    # Load the Stable Diffusion inpainting model with openpose_controlnet
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        inpaint_model_id, 
        controlnet=openpose_controlnet, 
        torch_dtype=torch.float16, 
        safety_checker=None,  
        requires_safety_checker=False,
        local_files_only=False)

    for lora_model_path in lora_model_paths:
        pipe.load_lora_weights(f"Models/Lora/{lora_model_path}")

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
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=sampling_steps,
            generator=generator,
            cross_attention_kwargs={"scale": lora_strength},
            image=scaled_image,
            width=generation_width,
            height=generation_height,
            mask_image=mask_image,
            guidance_scale=cfg_scale,
            control_image=pose_image,
            
            
        ).images[0]

        #output_image = fix_hands(output_image)

        # Save the generated image
        output_image.save(f"{save_folder_path}/{i + 1}:generated_image.jpg")

        composite_image = combine_images(image, output_image, mask)
        composite_image.save(f"{save_folder_path}/{i + 1}:composite_image.jpg")

        output_image = paste_image(input_image, composite_image, top_left)
        output_image = Image.fromarray(output_image)
        output_image.save(f"{save_folder_path}/{i + 1}:output_image.jpg")

        output_images.append(output_image)

    return output_images
