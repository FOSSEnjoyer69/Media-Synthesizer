import cv2
import numpy as np
import gradio as gr
import math
from icecream import ic
from image_editing import scale_image
from image_border import add_equal_image_border, remove_equal_image_border

from controlnet_aux import OpenposeDetector

stickwidth:int = 4

RIGHT_EAR_COLOUR =      [255, 0, 170]
LEFT_EAR_COLOUR =       [255, 0, 85]
RIGHT_EYE_COLOUR =      [170, 0, 255]
LEFT_EYE_COLOUR =       [255, 0, 255]
NOSE_COLOUR =           [255, 0, 0]
NECK_COLOUR =           [255, 85, 0]
RIGHT_SHOULDER_COLOUR = [255, 170, 0]
RIGHT_ELBOW_COLOUR =    [255, 255, 0]
RIGHT_WRIST_COLOUR =    [170, 255, 0]
LEFT_SHOULDER_COLOUR =  [85, 255, 0]
LEFT_ELBoW_COLOUR =     [0, 255, 0]
LEFT_WRIST_COLOUR =     [0, 255, 85]
RIGHT_HIP_COLOUR =      [0, 255, 170]
RIGHT_KNEE_COLOUR =     [0, 255, 170]
RIGHT_ANKLE_COLOUR =    [0, 170, 255]
LEFT_HIP_COLOUR =       [0, 85, 255]
LEFT_KNEE_COLOUR =      [0, 0, 255]
LEFT_ANKLE_COLOUR =     [85, 0, 255]


LIMB_SEQUENCE_COLOURS = [NOSE_COLOUR, NECK_COLOUR, RIGHT_SHOULDER_COLOUR, RIGHT_ELBOW_COLOUR, RIGHT_WRIST_COLOUR, LEFT_SHOULDER_COLOUR, LEFT_ELBoW_COLOUR, \
                            LEFT_WRIST_COLOUR, RIGHT_HIP_COLOUR, RIGHT_KNEE_COLOUR, RIGHT_ANKLE_COLOUR, LEFT_HIP_COLOUR, LEFT_KNEE_COLOUR, LEFT_ANKLE_COLOUR, \
                            RIGHT_EYE_COLOUR, LEFT_EYE_COLOUR,
                            RIGHT_EAR_COLOUR, LEFT_EAR_COLOUR]

LIMB_SEQUENCE = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]


JOINT_NAMES:list = ["Nose", "Left Eye", "Right Eye",
                    "Neck", 
                    "Left Shoulder", "Left Elbow", "Left Wrist",
                    "Right Shoulder", "Right Elbow", "Right Wrist",
                    "Left Hip", "Left Knee", "Left Ankle",
                    "Right Hip", "Right Knee", "Right Ankle"]


JOINT_NAMES_BY_COLOUR:dict = {
    (255, 0, 85): "Left Ear",
    (255, 0, 170): "Right Ear",
    (255, 0, 255): "Left Eye",
    (170, 0, 255): "Right Eye",
    (255, 0, 0): "Nose",
    (255, 85, 0): "Neck",
    (85, 255, 0): "Left Shoulder",
    (0, 255, 0): "Left Elbow",
    (0, 255, 85): "Left Wrist",
    (255, 170, 0): "Right Shoulder",
    (255, 255, 0): "Right Elbow",
    (170, 255, 0): "Right Wrist",
    (0, 255, 170): "Right Hip",
    (0, 255, 255): "Right Knee",
    (0, 170, 255): "Right Ankle",
    (0, 85, 255): "Left Hip",
    (0, 0, 255): "Left Knee",
    (85, 0, 255): "Left Ankle"}

JOINT_COLOURS:dict = {
    "Left Ear": (255, 0, 85),
    "Right Ear": (255, 0, 170),

    "Left Eye": (255, 0, 255),
    "Right Eye": (170, 0, 255),

    "Nose": (255, 0, 0),
    "Neck": (255, 85, 0),

    "Left Shoulder": (85, 255, 0),
    "Left Elbow": (0, 255, 0),
    "Left Wrist": (0, 255, 85),

    "Right Shoulder": (255, 170, 0),
    "Right Elbow": (255, 255, 0),
    "Right Wrist":(170, 255, 0),

    "Right Hip" : (0, 255, 170),
    "Right Knee" : (0, 255, 255),
    "Right Ankle" : (0, 170, 255),

    "Left Hip" : (0, 85, 255),
    "Left Knee" : (0, 0, 255),
    "Left Ankle" : (85, 0, 255),}


joint_radius:int = 4
pose_map_border_size:str = "5%"

def get_connected_joints(current_joint:str):
    if current_joint == "Left Ear":
        return "Left Eye"

    if current_joint == "Right Ear":
        return "Right Eye"

    elif current_joint == "Left Eye":
        return "Nose", "Left Ear"

    elif current_joint == "Right Eye":
        return "Nose", "Right Ear"

    elif current_joint == "Nose":
        return "Neck", "Left Eye", "Right Eye"

    elif current_joint == "Neck":
        return "Nose", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"

    #Left Arm
    elif current_joint == "Left Shoulder":
        return "Neck", "Left Elbow"
    elif current_joint == "Left Elbow":
        return "Left Shoulder", "Left Wrist"
    elif current_joint == "Left Wrist":
        return "Left Wrist"

    #Right Arm
    elif current_joint == "Right Shoulder":
        return "Neck", "Right Elbow"
    elif current_joint == "Right Elbow":
        return "Right Shoulder", "Right Wrist"
    elif current_joint == "Right Wrist":
        return "Right Elbow"

    #Left Leg
    elif current_joint == "Left Hip":
        return "Neck", "Left Knee"
    elif current_joint == "Left Knee":
        return "Left Hip", "Left Ankle"
    elif current_joint == "Left Ankle":
        return "Left Knee"

    #Right Leg
    elif current_joint == "Right Hip":
        return "Neck", "Right Knee"
    elif current_joint == "Right Knee":
        return "Right Hip", "Right Ankle"
    elif current_joint == "Right Ankle":
        return "Right Knee"

def get_connection_colours(joint_a_name:str, joint_b_name:str, multiplier:float=1):
    colour:tuple = (255, 255, 255)

    #Head
    if {joint_a_name, joint_b_name} == {"Neck", "Nose"}:
        colour = (0, 0, 153)
    if {joint_a_name, joint_b_name} == {"Nose", "Left Eye"}:
        colour = (153, 0, 153)
    if {joint_a_name, joint_b_name} == {"Left Eye", "Left Ear"}:
        colour = (153, 0, 102)
    if {joint_a_name, joint_b_name} == {"Nose", "Right Eye"}:
        colour = (51, 0, 153)
    if {joint_a_name, joint_b_name} == {"Right Eye", "Right Ear"}:
        colour = (102, 0, 153)

    #Left Arm
    if {joint_a_name, joint_b_name} == {"Neck", "Left Shoulder"}:
        colour = (153, 51, 0)
    if {joint_a_name, joint_b_name} == {"Left Shoulder", "Left Elbow"}:
        colour = (102, 153, 0)
    if {joint_a_name, joint_b_name} == {"Left Elbow", "Left Wrist"}:
        colour = (50, 154, 0)

    #Right Arm
    if {joint_a_name, joint_b_name} == {"Neck", "Right Shoulder"}:
        colour = (153, 0, 0)
    if {joint_a_name, joint_b_name} == {"Right Shoulder", "Right Elbow"}:
        colour = (150, 105, 1)
    if {joint_a_name, joint_b_name} == {"Right Elbow", "Right Wrist"}:
        colour = (152, 155, 4)

    #Left Leg
    if {joint_a_name, joint_b_name} == {"Neck", "Left Hip"}:
        colour = (0, 153, 153)
    if {joint_a_name, joint_b_name} == {"Left Hip", "Left Knee"}:
        colour = (0, 102, 153)
    if {joint_a_name, joint_b_name} == {"Left Knee", "Left Ankle"}:
        colour = (0, 51, 153)

    #Right Leg
    if {joint_a_name, joint_b_name} == {"Neck", "Right Hip"}:
        return (0, 153, 0)
    if {joint_a_name, joint_b_name} == {"Right Hip", "Right Knee"}:
        return (0, 153, 51)
    if {joint_a_name, joint_b_name} == {"Right Knee", "Right Ankle"}:
        return (0, 153, 102)

    if colour == (255, 255, 255):
        print(f"Could not find connection colours for {joint_a_name} & {joint_b_name}, returning white")
    
    colour = tuple(int(element * multiplier) for element in colour)


    return colour

def create_pose_image(image:np.ndarray, joints:dict, detection_padding:str="5%"):
    pose_map:np.ndarray = add_equal_image_border(image, pose_map_border_size)
    pose_image = np.zeros_like(image)

    height, width = pose_map.shape[:2]

    #Draw Bones
    for name, position in joints.items():
        person_index, joint = name.split("/")
        connected_joints = get_connected_joints(joint)
        for connected_joint in connected_joints:
            #Skip to next if connected joint is not being used
            if not joints.__contains__(f"{person_index}/{connected_joint}"):
                continue

            connection_colour = get_connection_colours(joint, connected_joint)
            
            bone_a_pos = position
            bone_b_pos = joints[f"{person_index}/{connected_joint}"]

            Y = np.array([bone_a_pos[0], bone_b_pos[0]])
            X = np.array([bone_a_pos[1], bone_b_pos[1]])

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

            cv2.fillConvexPoly(pose_map, polygon, [int(float(c) * 0.6) for c in connection_colour])
            cv2.fillConvexPoly(pose_image, polygon, [int(float(c) * 0.6) for c in connection_colour])

    #Draw Joints
    for name, position in joints.items():
        person_index, joint = name.split("/")
        joint_colour = JOINT_COLOURS[joint]

        cv2.circle(pose_map, position, joint_radius, joint_colour, thickness=-1)
        cv2.circle(pose_image, position, joint_radius, joint_colour, thickness=-1)

    #unbordered_pose_map = remove_equal_image_border(pose_map, pose_map_border_size)
    unbordered_pose_image = remove_equal_image_border(pose_image, pose_map_border_size)
    return pose_map, unbordered_pose_image

def detect_poses(image:np.ndarray, current_person_index, detection_resolution:int=512, detection_padding:str="5%") -> (np.ndarray, np.ndarray, dict):
    bordered_image = add_equal_image_border(image, detection_padding)
    height, width = bordered_image.shape[:2]

    detection_image = scale_image(bordered_image, detection_resolution)

    composite_image = bordered_image
    pose_image = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    joints = {}

    openpose:OpenposeDetector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir="huggingface cache")
    poses = openpose.detect_poses(detection_image)
    for index, pose in enumerate(poses):

        keypoints = pose.body.keypoints

        #Draw Bones
        for (k1_index, k2_index), color in zip(LIMB_SEQUENCE, LIMB_SEQUENCE_COLOURS):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1 is None or keypoint2 is None:
                continue

            bone_Position_Y = np.array([keypoint1.x, keypoint2.x]) * float(width)
            bone_Position_X = np.array([keypoint1.y, keypoint2.y]) * float(height)
            mX = np.mean(bone_Position_X)
            mY = np.mean(bone_Position_Y)
            length = ((bone_Position_X[0] - bone_Position_X[1]) ** 2 + (bone_Position_Y[0] - bone_Position_Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(bone_Position_X[0] - bone_Position_X[1], bone_Position_Y[0] - bone_Position_Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            colour = [int(float(c) * 0.6) for c in color] 
            composite_colour = tuple(int(element * 1 if (index + 1) == current_person_index else 0.5) for element in colour)

            cv2.fillConvexPoly(pose_image, polygon, colour)
            cv2.fillConvexPoly(composite_image, polygon, composite_colour)

        #Draw Joints
        for keypoint, color in zip(keypoints, LIMB_SEQUENCE_COLOURS):
            if keypoint is None:
                continue

            colour = tuple(int(element * 1 if (index + 1) == current_person_index else 0.5) for element in color)

            x, y = keypoint.x, keypoint.y
            x = int(x * width)
            y = int(y * height)

            cv2.circle(pose_image, (int(x), int(y)), 4, color, thickness=-1)
            cv2.circle(composite_image, (int(x), int(y)), 4, colour, thickness=-1)

            joint_name = JOINT_NAMES_BY_COLOUR[tuple(color)]
            joints[f"{index + 1}/{joint_name}"] = (x, y)

    return composite_image, remove_equal_image_border(pose_image, detection_padding), joints

def clear_poses(image:np.ndarray):
    return image, image, []

def remove_pose(image, current_person_index:int, joints:dict):
    keys_to_delete = []
    for key in joints:
        if key.startswith(str(current_person_index)):
            keys_to_delete.append(key)

    for key in keys_to_delete:
        joints.pop(key)
    
    pose_map, unbordered_pose_image = create_pose_image(image, joints)

    return pose_map, unbordered_pose_image, joints

def remove_joint(image:np.ndarray, current_person_index:int, current_joint_name:str, joints:dict):
    joint_key = f"{current_person_index}/{current_joint_name}"
    if joint_key in joints:
        joints.pop(joint_key)

    pose_map, unbordered_pose_image = create_pose_image(image, joints)

    return pose_map, unbordered_pose_image, joints

def pose_map_select(image:np.ndarray, current_person_index:int, current_joint_name:str, joints:dict, select:gr.SelectData):
    joints[f"{current_person_index}/{current_joint_name}"] = select.index #Set the new position of the current joint

    pose_map, unbordered_pose_image = create_pose_image(image, joints)

    return pose_map, unbordered_pose_image, joints
