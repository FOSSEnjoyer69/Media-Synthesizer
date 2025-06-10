#Text2Image
STABLE_DIFFUSION_TEXT2IMAGE_MODELS = ["CompVis/stable-diffusion-v1-4", "SG161222/Realistic_Vision_V6.0_B1_noVAE"]
STABLE_CASCADE_MODEL_IDS = ["stabilityai/stable-cascade-prior"]

TEXT_2_IMAGE_MODELS = STABLE_DIFFUSION_TEXT2IMAGE_MODELS + STABLE_CASCADE_MODEL_IDS

#Image2Image Inpaint
STABLE_DIFFUSION_INPAINT_MODELS = ["Uminosachi/realisticVisionV51_v51VAE-inpainting", "Lykon/absolute-reality-1.6525-inpainting", "Uminosachi/dreamshaper_8Inpainting", "runwayml/stable-diffusion-inpainting", "stabilityai/stable-diffusion-2-inpainting"]
IMAGE_INPAINT_MODELS = STABLE_DIFFUSION_INPAINT_MODELS