import gc
import torch
from diffusers import (StableCascadeDecoderPipeline, StableCascadePriorPipeline,
                        FluxPipeline)
import random
import numpy as np

import cv2

from save import get_output_folder

TEXT_2_IMAGE_MODEL_IDS = ["stabilityai/stable-cascade-prior", "black-forest-labs/FLUX.1-dev"]
STABLE_CASCADE_MODEL_IDS = ["stabilityai/stable-cascade-prior"]
FLUX_MODEL_IDS = ["black-forest-labs/FLUX.1-dev"]

def run_text2image(prompt:str, n_prompt:str,
                    model_id,
                    width:int=512, height=512,
                    sampling_steps:int=10,
                    cfg_scale:float=4, seed:int=-1,
                    iteration_count:int=1
                    ):
    if seed == -1:
        seed = random.randint(0, 2147483647)

    save_folder_path:str = get_output_folder()

    output:list = []
    if STABLE_CASCADE_MODEL_IDS.__contains__(model_id):
        output = run_stable_cascade(prompt, n_prompt, width, height, sampling_steps, cfg_scale, seed, iteration_count)
    elif FLUX_MODEL_IDS.__contains__(model_id):
        output = run_flux_text2image(prompt, n_prompt, width, height, cfg_scale, sampling_steps, seed, iteration_count, save_folder_path)
    else:
        output = np.zeros((width, height, 3), dtype=np.float16)

    return output

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

def run_flux_text2image(prompt:str, n_prompt:str,
                        width:int=512, height=512,
                        cfg_scale:float=3.5, sampling_steps:int=50, seed:int=-1,
                        iteration_count:int=1, save_folder_path:str=""
                        ):
    if seed == -1:
        seed = random.randint(0, 2147483647)

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir="huggingface cache")
    #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    #pipe.enable_xformers_memory_efficient_attention()

    output_list:list = []

    for i in range(iteration_count):
        image_seed = seed + i
        image = pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            width=width,
            height=height,
            guidance_scale=cfg_scale,
            num_inference_steps=sampling_steps,
            generator=torch.Generator("cpu").manual_seed(image_seed),
            output_type="np.array"
        )[0]

        if save_folder_path != "":
            cv2.imwrite(f"{save_folder_path}/{i}.jpg", image)

        output_list.append(image)

    return output_list