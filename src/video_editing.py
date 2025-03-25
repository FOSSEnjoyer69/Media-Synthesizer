def grow_video_mask(input_video_path:str, mask_video_path:str, expand_iteration:int=1):
    save_folder_path:str = get_output_folder()

    input_frames, fps = get_frames(input_video_path)
    mask_frames, _ = get_frames(mask_video_path)

    composite_video_path:str = f"{save_folder_path}/composite.mp4"
    mask_video_path:str = f"{save_folder_path}/mask.mp4"

    for i in range(len(input_frames)):
        input_frames[i], mask_frames[i] = grow_mask(input_frames[i], mask_frames[i], expand_iteration)

    save_video(input_frames, composite_video_path, fps)
    save_video(mask_frames, mask_video_path, fps)

    return composite_video_path, mask_video_path