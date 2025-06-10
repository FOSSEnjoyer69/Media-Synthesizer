from diffusers import DiffusionPipeline
import numpy as np
import random
import torch

from image_editing import auto_resize_to_pil, combine_images, RGBA_to_mask_RGB
from models import PAINT_BY_EXAMPLE_MODEL

def run_paint_by_example(image:np.ndarray, mask:np.ndarray, reference_image:np.ndarray, 
                         num_inference_steps:int=50, guidance_scale:float=7.5, 
                         seed:int=-1, iteration_count:int=1):
    pipe = DiffusionPipeline.from_pretrained(PAINT_BY_EXAMPLE_MODEL, torch_dtype=torch.float16, cache_dir="huggingface cache")
    pipe = pipe.to("cuda")

    if seed == -1:
        seed = random.randint(0, 2147483647)

    mask = RGBA_to_mask_RGB(mask["layers"][0])

    [image, mask, reference_image] = auto_resize_to_pil([image, mask, reference_image])

    width, height = image.size

    output_images = []
    for i in range(iteration_count):
        image_seed = seed + i
        output = pipe(
            image=image,
            mask_image=mask,
            example_image=reference_image,
            generator=torch.Generator('cuda').manual_seed(image_seed),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,

            width=width,
            height=height
        ).images[0]

        output = combine_images(image, output, mask, output_type="numpy")
        output_images.append(output)
    return output_images

