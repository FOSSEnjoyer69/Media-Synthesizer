from PIL import Image
import numpy as np

import gradio as gr

from prompt_presets import get_image_inpaint_preset_file_paths, get_image_inpaint_preset_file_names, load_prompts
from image_inpainting import run_image_inpaint as generate_inpaint_images, IMAGE_INPAINT_MODELS
from image_masking import add_mask, trim_mask, clear_mask, grow_mask, shrink_mask
from openpose_tools import detect_poses, pose_map_select, JOINT_NAMES, remove_pose

from text2image import run_text2image, TEXT_2_IMAGE_MODEL_IDS

from sam_tools import get_point_from_gradio_click, run_image_sam, MODELS as SAM_MODELS
from lora_manager import get_lora_model_names

IMAGE_INPAINT_PRESET_FILE_NAMES = get_image_inpaint_preset_file_names()


with gr.Blocks(title="Media Synthesizer", css=".app { max-width: 100% !important; }") as app:
    with gr.Tab("Create"):
        with gr.Tab("Text"):
            with gr.Tab("Text2Image"):
                with gr.Accordion("Prompts"):
                    with gr.Row():
                        text2image_prompt = gr.TextArea(label="Prompt", elem_id="text2image_prompt")
                        text2image_n_prompt = gr.TextArea(label="Negative Prompt", elem_id="text2image_n_prompt")
                with gr.Accordion("Generation Settings"):
                    text2image_model_id = gr.Dropdown(label="Model", choices=TEXT_2_IMAGE_MODEL_IDS, value=TEXT_2_IMAGE_MODEL_IDS[0])
                    text2image_sampling_steps_slider = gr.Slider(label="Sampling Steps", minimum=1, value=10, maximum=100)
                    with gr.Row():
                        text2image_cfg_slider = gr.Slider(label="CFG", minimum=0, value=4, maximum=10)
                        text2image_seed_slider = gr.Slider(label="Seed", minimum=-1, value=-1, maximum=pow(2, 32), step=1)

                    text2image_iteration_count = gr.Slider(label="Iteration Count", minimum=1, value=1, maximum=100, step=1)
                    with gr.Accordion("Resolution"):
                        text2image_width_slider = gr.Slider(label="Width", minimum=16, value=512, maximum=1024, step=1)
                        text2image_height_slider = gr.Slider(label="Height", minimum=16, value=512, maximum=1024, step=1)
                run_text2image_btn = gr.Button("Generate")
                text2image_output_gallery = gr.Gallery()

                run_text2image_btn.click(run_text2image, inputs=[text2image_prompt, text2image_n_prompt,
                                                                text2image_model_id,
                                                                text2image_width_slider, text2image_height_slider, 
                                                                text2image_sampling_steps_slider, 
                                                                text2image_cfg_slider, text2image_seed_slider,

                                                                text2image_iteration_count], 
                                                        outputs=[text2image_output_gallery])
        with gr.Tab("Image"):
            input_image = gr.Image()
            with gr.Tab("Inpaint"):
                with gr.Tab(label="Generate") as image_inpaint_generate_tab:
                    with gr.Row():
                        with gr.Accordion("Prompts"):
                            with gr.Row():
                                image_inpaint_prompt = gr.TextArea(label="Prompt", elem_id="image_inpaint_prompt")
                                image_inpaint_n_prompt = gr.TextArea(label="Negative Prompt", elem_id="image_inpaint_n_prompt")
                            image_inpaint_prompt_preset = gr.Dropdown(label="Prompts Preset", choices=IMAGE_INPAINT_PRESET_FILE_NAMES, value=IMAGE_INPAINT_PRESET_FILE_NAMES[0])
                            image_inpaint_prompt_preset.change(load_prompts, inputs=[image_inpaint_prompt_preset], outputs=[image_inpaint_prompt, image_inpaint_n_prompt])
                        with gr.Accordion("Loras"):
                            image_inpaint_lora_models = gr.Dropdown(label="Model", choices=get_lora_model_names(), multiselect=True)
                            image_inpaint_lora_strength = gr.Slider(label="Strangth", minimum=-1, value=0, maximum=1)

                    with gr.Accordion("Generation Settings"):
                        with gr.Row():
                            image_inpaint_sampling_steps = gr.Slider(label="Sampling Steps", minimum=1, value=60, maximum=100, step=1)
                            image_inpaint_cfg_scale = gr.Slider(label="CFG Scale", minimum=0, value=4, maximum=7, step=0.1)
                        image_inpaint_seed = gr.Slider(label="Seed", minimum=-1, value=-1, maximum=2147483647)
                        image_inpaint_iteration_count = gr.Slider(label="Iteration Count", minimum=1, value=2, maximum=100, step=1)
                        image_inpaint_inpaint_model_id = gr.Dropdown(label="Inpaint Model", choices=IMAGE_INPAINT_MODELS, value=IMAGE_INPAINT_MODELS[0])
                        with gr.Accordion("Generation Resolution"):
                            image_inpaint_min_generation_resolution = gr.Slider(label="Min Resolution", minimum=0, value=512, maximum=512)
                            image_inpaint_max_generation_resolution = gr.Slider(label="Max Resolution", minimum=512, value=1000, maximum=1000)
                    inpaint_images_btn = gr.Button("Generate")
                    image_inpaint_output_gallery = gr.Gallery()
                with gr.Tab(label="Mask") as image_inpaint_mask_tab:
                    with gr.Accordion("Segmant Anything"):
                        with gr.Row():
                            image_inpaint_sam_points = gr.State([])
                            image_inpaint_sam_labels = gr.State([])

                            image_inpaint_sam_map = gr.Image()
                            with gr.Column():
                                image_inpaint_run_same_btn = gr.Button("Run SAM")
                                with gr.Row():
                                    image_inpaint_sam_point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include", scale=2)
                                    image_inpaint_clear_sam_points = gr.Button("Clear Points")
                                    image_inpaint_sam_model = gr.Dropdown(label="SAM Model", choices=SAM_MODELS, value=SAM_MODELS[0])

                            image_inpaint_sam_map.select(get_point_from_gradio_click, inputs=[image_inpaint_sam_point_type, image_inpaint_sam_points, image_inpaint_sam_labels, image_inpaint_sam_map], outputs=[image_inpaint_sam_points, image_inpaint_sam_labels, image_inpaint_sam_map], queue=False)
                            image_inpaint_clear_sam_points.click(fn=lambda x: (x, [], []), inputs=[input_image], outputs=[image_inpaint_sam_map, image_inpaint_sam_points, image_inpaint_sam_labels])
                    with gr.Row():
                        with gr.Column():
                            composite_mask_image = gr.ImageEditor(label="Mask Composite", sources=(), brush=gr.Brush(colors=["#000000"], color_mode="fixed"), interactive=True)
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
                                current_pose_person_index_slider = gr.Slider(label="Current Person", minimum=1, maximum=10, step=1)
                                remove_pose_person_btn = gr.Button("Remove Person")
                            current_pose_joint = gr.Dropdown(label="Joint", value=JOINT_NAMES[0], choices=JOINT_NAMES)

                            with gr.Row():
                                composite_pose_image = gr.Image(label="Pose Composite")
                                pose_image = gr.Image(label="Pose")

                            detect_poses_btn.click(detect_poses, inputs=[input_image, current_pose_person_index_slider], outputs=[composite_pose_image, pose_image, pose_joints])
                            remove_pose_person_btn.click(remove_pose, inputs=[input_image, current_pose_person_index_slider, pose_joints], outputs=[composite_pose_image, pose_image, pose_joints])
                            composite_pose_image.select(pose_map_select, inputs=[input_image, current_pose_person_index_slider, current_pose_joint, pose_joints], outputs=[composite_pose_image, pose_image, pose_joints])
            with gr.Tab("Extractor"):
                image_extractor_selection = gr.Sketchpad(label="Mask Composite", interactive=True)



        input_image.upload(fn=lambda x: ([gr.update(value=x)]*4), inputs=[input_image], 
                                                   outputs=[input_image, 
                                                            
                                                            #Inpaint
                                                            composite_mask_image, image_inpaint_sam_map,
                                                            
                                                            #Extractor
                                                            image_extractor_selection
                                                            ])
        image_inpaint_run_same_btn.click(run_image_sam, inputs=[input_image, image_inpaint_sam_model, image_inpaint_sam_points, image_inpaint_sam_labels], outputs=[composite_mask_image, mask_image])
        
        inpaint_images_btn.click(generate_inpaint_images, inputs=[input_image, mask_image, pose_image, 
                                                                image_inpaint_prompt, image_inpaint_n_prompt, 
                                                                image_inpaint_sampling_steps, image_inpaint_cfg_scale, image_inpaint_seed, image_inpaint_iteration_count, 
                                                                image_inpaint_inpaint_model_id, 
                                                                image_inpaint_lora_models, image_inpaint_lora_strength, 
                                                                image_inpaint_min_generation_resolution, image_inpaint_max_generation_resolution,
                                                                ], 
                                                            outputs=[image_inpaint_output_gallery])
    with gr.Tab("Settings"):
        with gr.Tab("Prompts"):
            with gr.Tab("Presets"):
                with gr.Accordion("Image Inpainting"):
                    imgage_inpaint_prompt_preset_files = gr.File(label="Preset Files", file_count="multiple", type="filepath", value=get_image_inpaint_preset_file_paths())
app.launch()