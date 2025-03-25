from PIL import Image
import numpy as np
import cv2

def get_gradio_sketchpad_mask(Sketchpad_output:dict):
    layer = Sketchpad_output["layers"][0]
    return RGBA_to_mask_RGB(layer)

def create_black_copy_of_image(image):
    height, width = image.shape[:2]
    black_copy = np.zeros((height, width, 3), dtype=np.uint8)
    return black_copy

def RGBA_to_mask_RGB(rgba_image):
    """
    Converts an RGBA image to a mask RGB image using the alpha channel as the mask.

    Parameters:
    - rgba_image: A NumPy array representing the RGBA image.

    Returns:
    - mask_rgb: A NumPy array representing the RGB mask image.
    """
    # Ensure the input is a NumPy array
    if isinstance(rgba_image, Image.Image):
        rgba_image = np.array(rgba_image)
    
    # Extract the alpha channel (4th channel in RGBA)
    alpha_channel = rgba_image[:, :, 3]

    # Create a binary mask from the alpha channel (values: 0 or 255)
    mask_binary = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

    # Stack the binary mask into 3 channels to form an RGB image
    mask_rgb = np.stack([mask_binary] * 3, axis=-1)

    return mask_rgb

def add_mask(input_image:np.ndarray, sel_mask, mask_image:np.ndarray, source:str="gradio"):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    if source == "gradio":
        selection_mask = get_gradio_sketchpad_mask(sel_mask)
    else:
        selection_mask = sel_mask


    mask_height, mask_width = mask_image.shape[:2]
    composite_height, composite_width = selection_mask.shape[:2]

    composite_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)
    scaled_up_selection_mask = cv2.resize(selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

    # Invert and combine masks
    inverted_mask = cv2.bitwise_not(mask_image)
    combined_mask = cv2.bitwise_or(mask_image, scaled_up_selection_mask & inverted_mask)

    scaled_down_mask = cv2.resize(combined_mask, (composite_width, composite_height), interpolation=cv2.INTER_NEAREST)

    # Blend images
    composite_image = cv2.addWeighted(composite_image, 0.5, scaled_down_mask, 0.5, 0)

    return composite_image, combined_mask

def trim_mask(input_image:np.ndarray, sel_mask, mask_image:np.ndarray, source:str="gradio"):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    if source == "gradio":
        selection_mask = get_gradio_sketchpad_mask(sel_mask)
    else:
        selection_mask = sel_mask


    mask_height, mask_width = mask_image.shape[:2]
    composite_height, composite_width = selection_mask.shape[:2]

    composite_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)
    scaled_up_selection_mask = cv2.resize(selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

    # Invert and combine masks
    #inverted_mask = cv2.bitwise_not(mask_image)
    combined_mask = cv2.bitwise_xor(mask_image, scaled_up_selection_mask & mask_image)

    scaled_down_mask = cv2.resize(combined_mask, (composite_width, composite_height), interpolation=cv2.INTER_NEAREST)

    # Blend images
    composite_image = cv2.addWeighted(composite_image, 0.5, scaled_down_mask, 0.5, 0)

    return composite_image, combined_mask

def grow_mask(input_image, mask_image:np.ndarray, expand_iteration:int=1):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    height, weight = mask_image.shape[:2]

    expand_iteration = int(np.clip(expand_iteration, 1, 100))
    new_mask = cv2.dilate(mask_image, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    composite_image = cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)

    return composite_image, new_mask

def shrink_mask(input_image, mask_image, expand_iteration: int = 1):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)


    expand_iteration:int = int(np.clip(expand_iteration, 1, 100))
    new_mask:np.ndarray = cv2.erode(mask_image, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    composite_image = cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)

    return composite_image, new_mask

def clear_mask(input_image):
    return input_image, create_black_copy_of_image(input_image)