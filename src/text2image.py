import gc
import torch
from diffusers import (StableDiffusionPipeline,
                       StableCascadeDecoderPipeline, StableCascadePriorPipeline)
import random
import torch
import numpy as np

import torch

from save import get_output_folder

from models import *

def run_text2image(prompt:str, n_prompt:str,
                    model_id,
                    width:int=512, height=512, 
                    sampling_steps:int=10,
                    cfg_scale:float=4, seed:int=-1,
                    iteration_count:int=1
                    ):


    if seed == -1:
        seed = random.randint(0, 2147483647)

    if model_id in STABLE_DIFFUSION_TEXT2IMAGE_MODELS:
        return run_stable_diffusion_v1(prompt, n_prompt, width, height, model_id, sampling_steps, cfg_scale, seed, iteration_count)
    elif model_id in STABLE_CASCADE_MODEL_IDS:
        return run_stable_cascade(prompt, n_prompt, width, height, sampling_steps, cfg_scale, seed, iteration_count)
    else:
        return np.zeros((width, height, 3), dtype=np.float16)

def run_stable_diffusion_v1(prompt:str, n_prompt:str,
                    width:int=512, height=512,
                    model:str="CompVis/stable-diffusion-v1-4", sampling_steps:int=10,
                    cfg_scale:float=4, seed:int=-1,
                    iteration_count:int=1
                    ):
    pipe = StableDiffusionPipeline.from_pretrained(model, 
                                                   torch_dtype=torch.float16, 
                                                   cache_dir="huggingface cache",
                                                   safety_checker=None,  
                                                   requires_safety_checker=False,
                                                   local_files_only=False,).to("cuda")

    output_list = []
    for i in range(iteration_count):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)

        output = pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            width=width,
            height=height,
            num_inference_steps=sampling_steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]

        output_list.append(output)
    return output_list


def run_stable_cascade(prompt:str, n_prompt:str,
                    width:int=512, height=512,
                    sampling_steps:int=10,
                    cfg_scale:float=4, seed:int=-1,
                    iteration_count:int=1
                    ):
    if seed == -1:
        seed = random.randint(0, 2147483647)

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16, cache_dir="huggingface cache")
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16, cache_dir="huggingface cache")

    prior.enable_model_cpu_offload()
    prior.enable_xformers_memory_efficient_attention()

    decoder.enable_model_cpu_offload()
    decoder.enable_xformers_memory_efficient_attention

    prior_output = prior(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=n_prompt,
        guidance_scale=cfg_scale,
        num_images_per_prompt=1,
        num_inference_steps=20
    )

    output_list = []
    for i in range(iteration_count):
        image_seed:int = seed + i
        gc.collect()

        generator = torch.Generator(device="cpu").manual_seed(image_seed)

        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings.to(torch.float16),
            prompt=prompt,
            negative_prompt=n_prompt,
            guidance_scale=cfg_scale,
            generator=generator,
            num_inference_steps=sampling_steps
        ).images[0]

        output_list.append(decoder_output)

    #Cleanup
    del prior
    del decoder
    torch.cuda.empty_cache()
    gc.collect()

    return output_list

