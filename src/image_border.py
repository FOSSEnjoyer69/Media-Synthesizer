import numpy as np
import cv2

def add_equal_image_border(image:np.ndarray, border_size:str="5%", border_colour:tuple = (0,0,0)):
    original_height, original_width = image.shape[:2]

    if border_size[-1] == '%':
        percentage = float(border_size[:-1])
        top_bottom_border_size = int(original_height * (percentage / 100))
        left_right_border_size = int(original_width * (percentage / 100))

    bordered_image = cv2.copyMakeBorder(
        image,
        top=top_bottom_border_size,
        bottom=top_bottom_border_size,
        left=left_right_border_size,
        right=left_right_border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=border_colour
    )

    return bordered_image

def remove_equal_image_border(image:np.ndarray, border_size:str="5%"):
    original_height, original_width = image.shape[:2]

    if border_size[-1] == '%':
        # Calculate border size as a percentage
        percentage = float(border_size[:-1])
        top_bottom_border_size = int(original_height * (percentage / 100))
        left_right_border_size = int(original_width * (percentage / 100))

    # Crop the border using array slicing
    cropped_image = image[
        top_bottom_border_size:original_height - top_bottom_border_size,
        left_right_border_size:original_width - left_right_border_size
    ]

    return cropped_image
