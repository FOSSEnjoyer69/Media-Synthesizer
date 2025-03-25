from torchvision import transforms

import cv2
import numpy as np
import gradio as gr

from icecream import ic

from PIL import Image, ImageFilter, ImageDraw

from save import get_output_folder

def set_mask_composite_image(image):
    if isinstance(image, dict):
        image = image["background"]

    image = scale_image(image, target_resolution=1024)
    height, width = image.shape[:2]

    return gr.update(value=image, height=height + int(height * 0.5))

def combine_images(inpaint_input_image, inpaint_image, inpaint_mask) -> Image.Image:
    if isinstance(inpaint_input_image, np.ndarray):
        inpaint_input_image = Image.fromarray(inpaint_input_image, 'RGB')

    if isinstance(inpaint_mask, np.ndarray):
        inpaint_mask = Image.fromarray(inpaint_mask, 'RGB')  # Ensure mask is in grayscale mode

    width, height = inpaint_input_image.size

    #Scale inpaint image and mask up to original resolution
    inpaint_image = inpaint_image.resize((width, height))
    inpaint_mask = inpaint_mask.resize((width, height))

    dilate_mask_image = Image.fromarray(cv2.dilate(np.array(inpaint_mask), np.ones((3, 3), dtype=np.uint8), iterations=4))
    output_image = Image.composite(inpaint_image, inpaint_input_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

    return output_image

def scale_image(image, target_resolution: int = 1000, allow_upscale: bool = True):
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.uint8)

    original_height, original_width = image.shape[:2]

    # Determine if width or height is the limiting factor
    if original_width >= original_height:
        # Scale based on width
        scale_factor = target_resolution / original_width
    else:
        # Scale based on height
        scale_factor = target_resolution / original_height

    # Prevent upscaling if allow_upscale is False and the image is already larger than the target
    if not allow_upscale and scale_factor > 1:
        return image  # Return the original image without upscaling

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Scale the image to the new dimensions
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return image

def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: float = 0.1, min_resolution: int = 512) -> np.ndarray:
    """
    Crops an image based on mask and padding, ensuring a minimum resolution.

    Parameters:
    image (numpy image): Image to crop.
    mask (numpy image): Crop reference.
    padding (float 0.0 to 1): Crop padding.
    min_resolution (tuple): Minimum resolution (height, width) for the crop.

    Returns:
    numpy image: Cropped image.
    """

    min_resolution = (min_resolution, min_resolution)

    # Resize the image to match the mask size if they are different
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Ensure the mask is a binary mask (non-zero regions define the crop)
    non_zero_points = np.argwhere(mask > 0)

    # Check if the mask is 2D (grayscale) or 3D (color image)
    if non_zero_points.shape[1] == 3:
        # For 3D masks (height, width, channels)
        y_min, x_min = np.min(non_zero_points[:, :2], axis=0)
        y_max, x_max = np.max(non_zero_points[:, :2], axis=0)
    else:
        # For 2D masks (height, width)
        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)

    # Compute padding in terms of pixels
    height, width = image.shape[:2]
    pad_x = int(padding * (x_max - x_min))
    pad_y = int(padding * (y_max - y_min))

    # Apply padding and ensure it stays within the image bounds
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)

    # Ensure the cropped region meets the minimum resolution
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    min_height, min_width = min_resolution
    if crop_width < min_width:
        expand_x = (min_width - crop_width) // 2
        x_min = max(0, x_min - expand_x)
        x_max = min(width, x_max + expand_x)
    if crop_height < min_height:
        expand_y = (min_height - crop_height) // 2
        y_min = max(0, y_min - expand_y)
        y_max = min(height, y_max + expand_y)

    # Crop the image using the adjusted bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

def auto_resize_to_pil(images):
    outputs = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

            width, height = image.size
            new_height = (height // 8) * 8
            new_width = (width // 8) * 8

            if new_width < width or new_height < height:
                if (new_width / width) < (new_height / height):
                    scale = new_height / height
                else:
                    scale = new_width / width
                resize_height = int(height*scale+0.5)
                resize_width = int(width*scale+0.5)
                if height != resize_height or width != resize_width:
                    image = transforms.functional.resize(image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
                if resize_height != new_height or resize_width != new_width:
                    image = transforms.functional.center_crop(image, (new_height, new_width))

            outputs.append(image)
        else:
            raise Exception(f"image of type {type(image)} is not supported, supported types are numpy")
    return outputs

def find_image_in_large_image(large_image: np.ndarray, cropped_image: np.ndarray):
    """
    Finds the coordinates of the cropped image within the large image.
    
    Args:
        large_image (np.ndarray): The large image as a NumPy array.
        cropped_image (np.ndarray): The cropped image as a NumPy array.

    Returns:
        Tuple of top-left and bottom-right coordinates of the found region.
    """
    # Convert images to grayscale for matching (optional)
    large_image_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(large_image_gray, cropped_image_gray, cv2.TM_CCOEFF_NORMED)

    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # The top-left corner of the matched area in the large image
    top_left = max_loc
    # The size of the cropped image (height and width)
    h, w = cropped_image_gray.shape[:2]
    # The bottom-right corner of the matched area
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Return the coordinates of the top-left and bottom-right corners
    return top_left, bottom_right

def paste_image(large_image: np.ndarray, image_to_paste: np.ndarray, top_left: tuple):
    """
    Pastes the image_to_paste onto the large_image at the specified coordinates.

    Args:
        large_image (np.ndarray): The large image as a NumPy array.
        image_to_paste (np.ndarray): The image to be pasted as a NumPy array.
        top_left (tuple): The coordinates of the top-left corner where the image will be pasted.

    Returns:
        np.ndarray: The large image with the pasted image.
    """
    if isinstance(image_to_paste, Image.Image):
        image_to_paste = np.array(image_to_paste)

    # Get the dimensions of the image to paste
    h, w = image_to_paste.shape[:2]

    # Extract the coordinates
    x, y = top_left

    # Ensure the image to paste fits within the large image
    if large_image.shape[0] < y + h or large_image.shape[1] < x + w:
        raise ValueError("Image to paste is too large for the target location in the large image.")

    # Paste the image (copy the image to paste into the region of interest)
    large_image[y:y+h, x:x+w] = image_to_paste

    # Return the modified large image
    return large_image


def create_borderd_image(image, outpaint_up:int, outpaint_down:int, outpaint_left:int, outpaint_right:int, overlap:int = 10):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, 'RGB')

    # Calculate new size
    new_width = image.width + outpaint_left + outpaint_right
    new_height = image.height + outpaint_up + outpaint_down

    # Create a new image with the new size
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    
    # Paste the original image onto the new image, centered within the new borders
    new_image.paste(image, (outpaint_left, outpaint_up))

    # Create a mask with the same size as the new image
    mask = Image.new("L", (new_width, new_height), 0)
    draw = ImageDraw.Draw(mask)

    # Add black borders and update the mask with overlap
    if outpaint_up > 0:
        black_border_top = Image.new("RGB", (new_width, outpaint_up), (0, 0, 0))
        new_image.paste(black_border_top, (0, 0))
        draw.rectangle([0, 0, new_width, outpaint_up + overlap], fill=255)
    
    if outpaint_down > 0:
        black_border_bottom = Image.new("RGB", (new_width, outpaint_down), (0, 0, 0))
        new_image.paste(black_border_bottom, (0, new_height - outpaint_down))
        draw.rectangle([0, new_height - outpaint_down - overlap, new_width, new_height], fill=255)
    
    if outpaint_left > 0:
        black_border_left = Image.new("RGB", (outpaint_left, new_height), (0, 0, 0))
        new_image.paste(black_border_left, (0, 0))
        draw.rectangle([0, 0, outpaint_left + overlap, new_height], fill=255)
    
    if outpaint_right > 0:
        black_border_right = Image.new("RGB", (outpaint_right, new_height), (0, 0, 0))
        new_image.paste(black_border_right, (new_width - outpaint_right, 0))
        draw.rectangle([new_width - outpaint_right - overlap, 0, new_width, new_height], fill=255)
    
    return new_image, mask

def add_mask(input_image, sel_mask, mask_image):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    mask_height, mask_width = mask_image.shape[:2]

    # Convert RGBA to binary mask
    selection_mask = RGBA_to_mask_RGB(sel_mask["layers"][0])
    composite_height, composite_width = selection_mask.shape[:2]

    # Resize input image to match selection mask size
    composite_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    # Scale up selection mask to match mask image size
    scaled_up_selection_mask = cv2.resize(selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

    # Invert and combine masks
    inverted_mask = cv2.bitwise_not(mask_image)
    combined_mask = cv2.bitwise_or(mask_image, scaled_up_selection_mask & inverted_mask)

    # Resize combined mask to match composite image size
    scaled_down_mask = cv2.resize(combined_mask, (composite_width, composite_height), interpolation=cv2.INTER_NEAREST)

    # Blend images
    composite_image = cv2.addWeighted(composite_image, 0.5, scaled_down_mask, 0.5, 0)

    return composite_image, combined_mask

def trim_mask(input_image, sel_mask, mask_image):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    mask_height, mask_width = mask_image.shape[:2]

    # Convert RGBA to binary mask and invert the selection
    selection_mask = np.logical_not(RGBA_to_mask_RGB(sel_mask["layers"][0]))
    selection_mask = selection_mask.astype(np.uint8) * 255  # Convert to 0-255 for compatibility

    composite_height, composite_width = selection_mask.shape[:2]

    # Resize input image to match selection mask size
    composite_image = cv2.resize(input_image, (composite_width, composite_height), interpolation=cv2.INTER_LINEAR)

    # Scale up the selection mask to match the mask image size
    scaled_up_selection_mask = cv2.resize(selection_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

    # Apply the selection mask to trim the existing mask
    trimmed_mask = cv2.bitwise_and(mask_image, scaled_up_selection_mask)

    # Resize the trimmed mask to match the composite image size
    scaled_down_mask = cv2.resize(trimmed_mask, (composite_width, composite_height), interpolation=cv2.INTER_NEAREST)

    # Blend the composite image with the scaled-down trimmed mask
    composite_image = cv2.addWeighted(composite_image, 0.5, scaled_down_mask, 0.5, 0)

    return composite_image, trimmed_mask

def grow_mask(input_image, mask_image:np.ndarray, expand_iteration:int=1):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)

    height, weight = mask_image.shape[:2]

    expand_iteration = int(np.clip(expand_iteration, 1, 100))
    new_mask = cv2.dilate(mask_image, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    composite_image = cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)
    composite_image = scale_image(composite_image, target_resolution=2048, allow_upscale=False)

    return composite_image, new_mask
    
def shrink_mask(input_image, mask_image, expand_iteration: int = 1):
    if mask_image is None:
        mask_image = create_black_copy_of_image(input_image)
        

    expand_iteration:int = int(np.clip(expand_iteration, 1, 100))
    new_mask:np.ndarray = cv2.erode(mask_image, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)


    composite_image = cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)
    composite_image = scale_image(composite_image, target_resolution=2048, allow_upscale=False)

    return composite_image, new_mask

def clear_mask(input_image):
    height, width, channels = input_image.shape

    scaled_input_image = scale_image(input_image, target_resolution=2048, allow_upscale=False)
    scaled_height, scale_width = scaled_input_image.shape[:2]
    
    mask_image = Image.new("RGB", (width, height), "black")

    return gr.update(value=scaled_input_image), gr.update(value=mask_image)

def create_black_copy_of_image(image):
    height, width = image.shape[:2]
    black_copy = np.zeros((height, width), dtype=np.uint8)
    return black_copy

def RGB_to_RGBA(rgb_image:np.ndarray):
    alpha_channel = np.full((rgb_image.shape[0], rgb_image.shape[1], 1), 255, dtype=np.uint8)
    rgba_image = np.concatenate((rgb_image, alpha_channel), axis=-1)
    return rgba_image

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