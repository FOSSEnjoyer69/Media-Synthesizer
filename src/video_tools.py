import cv2
import numpy as np

def save_video(frames, output_filename, fps=30, codec='MP4V'):
    """
    Save a sequence of frames as a video.

    Args:
        frames (list): List of frames (numpy arrays) to save as a video.
        output_filename (str): Output video file name (e.g., "output.avi").
        fps (int): Frames per second for the video. Default is 30.
        codec (str): FourCC codec code for the video (e.g., "XVID", "MP4V"). Default is "XVID".
    
    Returns:
        None
    """
    if not frames:
        raise ValueError("Frame list is empty. Provide at least one frame.")

    # Get the size of the first frame
    height, width, channels = frames[0].shape
    frame_size = (width, height)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # Write each frame to the video
    for frame in frames:
        if frame.shape[0:2] != (height, width):
            raise ValueError("All frames must have the same dimensions.")
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved successfully as '{output_filename}'")

def composite_video_mask(base_video_frames, mask_video_frames, output_path:str, fps:int, codec:str='MP4V'):
    height, width, channels = base_video_frames[0].shape
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for i in range(len(base_video_frames)):
        composite_frame = cv2.addWeighted(base_video_frames[i], 0.5, mask_video_frames[i], 0.5, 0)

        video_writer.write(cv2.cvtColor(composite_frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    return output_path        

def get_frames(video_path:str) -> (list, int):
    vid_cap = cv2.VideoCapture(video_path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    frames = []

    frame_count = 0
    while True:
        ret, frame = vid_cap.read()

        if not ret:
            break

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_count += 1


    return frames, fps