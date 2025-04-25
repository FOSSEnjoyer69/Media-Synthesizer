import numpy as np
import cv2
import torch
from PIL import Image

import gradio as gr 

from sam2.sam2_image_predictor import SAM2ImagePredictor

MODELS = ["facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-small", "facebook/sam2.1-hiera-base-plus", "facebook/sam2.1-hiera-large"]

def get_point_from_gradio_click(point_type, tracking_points, trackings_input_label, first_frame, evt: gr.SelectData):
    if isinstance(first_frame, np.ndarray):
        first_frame:Image = Image.fromarray(first_frame)

    if isinstance(tracking_points, gr.State):
        tracking_points = []  # Reset to an empty list if it was incorrectly set

    if isinstance(trackings_input_label, gr.State):
        trackings_input_label = []

    tracking_points.append(evt.index)

    if point_type == "include":
        trackings_input_label.append(1)
    elif point_type == "exclude":
        trackings_input_label.append(0)

    transparent_background = first_frame.convert('RGBA')
    width, height = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.02
    radius = int(fraction * min(width, height))

    transparent_layer = np.zeros((height, width, 4), dtype=np.uint8)
    
    for index, track in enumerate(tracking_points):
        if trackings_input_label[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)

    return tracking_points, trackings_input_label, np.array(selected_point_map)

def run_image_sam(image:np.ndarray, model_id:str, points:list, labels:list):
    mask_height, mask_width = image.shape[:2]
    
    #image = scale_image(image, target_resolution=2048, allow_upscale=False)
    mask_image = np.zeros_like(image, dtype=np.uint8)

    if (points == [] or points is None) or (labels == [] or labels is None):
        return gr.update(value=image), gr.update(value=mask_image)

    predictor = SAM2ImagePredictor.from_pretrained(model_id, cache_dir="huggingface cache")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels)

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_layer = (mask > 0).astype(np.uint8) * 255
            for c in range(3):  # Assuming RGB, repeat mask for all channels
                mask_image[:, :, c] = mask_layer
    

    compisite_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    mask_image = cv2.resize(mask_image, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)

    return compisite_image, mask_image
