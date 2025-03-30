from PIL import Image
import numpy as np

import gradio as gr

from prompt_presets import get_image_inpaint_preset_file_paths, get_image_inpaint_preset_file_names, load_prompts
from image_inpainting import generate_inpaint_images
from image_masking import add_mask, trim_mask, clear_mask, grow_mask, shrink_mask
from openpose_tools import detect_poses, pose_map_select
from sam_tools import get_point_from_gradio_click, run_image_sam, MODELS as SAM_MODELS

IMAGE_INPAINT_MODELS = ["Uminosachi/realisticVisionV51_v51VAE-inpainting", "Lykon/absolute-reality-1.6525-inpainting", "Uminosachi/dreamshaper_8Inpainting"]

def input_file_changed(new_file):
    image = None

    try:
        image = np.array(Image.open(new_file))
    except:
        image = None

    is_image_mode:bool = image is not None

    return gr.update(visible=is_image_mode), gr.update(value=image, visible=is_image_mode), gr.update(value=image), gr.update(value=image)

with gr.Blocks(title="Media Synthesizer", css=".app { max-width: 100% !important; }") as app:
    with gr.Tab("Create"):
        input_file = gr.File(file_count="single", height=250)
        with gr.Row():
            input_image = gr.Image(interactive=False, visible=False)

        with gr.Tab(label="", visible=False) as image_tab:
            with gr.Tab(label="Inpaint") as image_inpaint_tab:
                with gr.Tab(label="Generate") as image_inpaint_generate_tab:
                    with gr.Accordion("Prompts"):
                        with gr.Row():
                            image_inpaint_prompt = gr.TextArea(label="Prompt", elem_id="image_inpaint_prompt")
                            image_inpaint_n_prompt = gr.TextArea(label="Negative Prompt", elem_id="image_inpaint_n_prompt")
                        image_inpaint_prompt_preset = gr.Dropdown(label="Prompts Preset", choices=get_image_inpaint_preset_file_names(), value=0)
                        image_inpaint_prompt_preset.change(load_prompts, inputs=[image_inpaint_prompt_preset], outputs=[image_inpaint_prompt, image_inpaint_n_prompt])
                    with gr.Accordion("Generation Settings"):
                        with gr.Row():
                            image_inpaint_sampling_steps = gr.Slider(label="Sampling Steps", minimum=1, value=60, maximum=100, step=1)
                            image_inpaint_cfg_scale = gr.Slider(label="CFG Scale", minimum=0, value=4, maximum=7, step=0.1)
                        image_inpaint_seed = gr.Slider(label="Seed", minimum=-1, value=-1, maximum=2147483647)
                        image_inpaint_iteration_count = gr.Slider(label="Iteration Count", minimum=1, value=2, maximum=100, step=1)
                        image_inpaint_inpaint_model_id = gr.Dropdown(label="Inpaint Model", choices=IMAGE_INPAINT_MODELS, value=IMAGE_INPAINT_MODELS[0])
                        with gr.Accordion("Generation Resolution"):
                            image_inpaint_min_generation_resolution = gr.Slider(label="Min Resolution", minimum=0, value=512, maximum=512)
                            image_inpaint_max_generation_resolution = gr.Slider(label="Max Resolution", minimum=512, value=512, maximum=1000)
                    inpaint_images_btn = gr.Button("Generate")
                    image_inpaint_output_gallery = gr.Gallery()

                with gr.Tab(label="Mask") as image_inpaint_mask_tab:
                    with gr.Accordion("Segmant Anything"):
                        with gr.Row():
                            image_inpaint_sam_points = gr.State([])
                            image_inpaint_sam_labels = gr.State([])

                            image_inpaint_sam_map = gr.Image()
                            image_inpaint_run_same_btn = gr.Button("Run SAM")
                            with gr.Column():
                                with gr.Row():
                                    image_inpaint_sam_point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include", scale=2)
                                    image_inpaint_clear_sam_points = gr.Button("Clear Points")
                                    image_inpaint_sam_model = gr.Dropdown(label="SAM Model", choices=SAM_MODELS, value=SAM_MODELS[0])

                            image_inpaint_sam_map.select(get_point_from_gradio_click, inputs=[image_inpaint_sam_point_type, image_inpaint_sam_points, image_inpaint_sam_labels, image_inpaint_sam_map], outputs=[image_inpaint_sam_points, image_inpaint_sam_labels, image_inpaint_sam_map], queue=False)
                            image_inpaint_clear_sam_points.click(fn=lambda x: (x, [], []), inputs=[input_image], outputs=[image_inpaint_sam_map, image_inpaint_sam_points, image_inpaint_sam_labels])
                    with gr.Row():
                        with gr.Column():
                            composite_mask_image = gr.Sketchpad(label="Mask Composite", interactive=True)
                            with gr.Row():
                                add_mask_btn = gr.Button("+")
                                trim_mask_btn = gr.Button("-")
                                clear_mask_btn = gr.Button("Clear")

                                shrink_mask_btn = gr.Button("Shrink")
                                shrink_grow_mask_amount_slider = gr.Slider(show_label=False, minimum=1, maximum=100, step=1, min_width=768)
                                grow_mask_btn = gr.Button("Grow")
                        with gr.Column():
                            mask_image = gr.Image(label="Mask")

                    add_mask_btn.click(add_mask, inputs=[input_image, composite_mask_image, mask_image], outputs=[composite_mask_image, mask_image])
                    trim_mask_btn.click(trim_mask, inputs=[input_image, composite_mask_image, mask_image], outputs=[composite_mask_image, mask_image])
                    clear_mask_btn.click(clear_mask, inputs=[input_image], outputs=[composite_mask_image, mask_image])

                    shrink_mask_btn.click(shrink_mask, inputs=[input_image, mask_image, shrink_grow_mask_amount_slider], outputs=[composite_mask_image, mask_image])
                    grow_mask_btn.click(grow_mask, inputs=[input_image, mask_image, shrink_grow_mask_amount_slider], outputs=[composite_mask_image, mask_image])
                with gr.Tab(label="Pose") as image_inpaint_pose_tab:
                    pose_joints = gr.State({})
                    with gr.Row():
                        with gr.Column():
                            detect_poses_btn = gr.Button("Detect Poses")
                            with gr.Row():
                                composite_pose_image = gr.Image(label="Pose Composite")
                                pose_image = gr.Image(label="Pose")

                            detect_poses_btn.click(detect_poses, inputs=[input_image], outputs=[composite_pose_image, pose_image, pose_joints])

        input_file.change(input_file_changed, inputs=[input_file], outputs=[image_tab, input_image, composite_mask_image, image_inpaint_sam_map])
        image_inpaint_run_same_btn.click(run_image_sam, inputs=[input_image, image_inpaint_sam_model, image_inpaint_sam_points, image_inpaint_sam_labels], outputs=[composite_mask_image, mask_image])
        
        inpaint_images_btn.click(generate_inpaint_images, inputs=[input_image, mask_image, pose_image, image_inpaint_prompt, image_inpaint_n_prompt, image_inpaint_sampling_steps, image_inpaint_cfg_scale, image_inpaint_seed, image_inpaint_iteration_count, image_inpaint_inpaint_model_id, image_inpaint_min_generation_resolution, image_inpaint_max_generation_resolution], outputs=[image_inpaint_output_gallery])
    with gr.Tab("Settings"):
        with gr.Tab("Prompts"):
            with gr.Tab("Presets"):
                with gr.Accordion("Image Inpainting"):
                    imgage_inpaint_prompt_preset_files = gr.File(label="Preset Files", file_count="multiple", type="filepath", value=get_image_inpaint_preset_file_paths())
app.launch()