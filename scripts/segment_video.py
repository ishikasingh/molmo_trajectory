import os
import colorsys
from pathlib import Path
import click
import cv2  # Replace moviepy with OpenCV
from PIL import Image
import numpy as np
import smart_open
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
import decord  # don't move this import earlier

device = torch.device("cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# model_name = "./checkpoints/sam2.1_hiera_large.pt"
model_name = "facebook/sam2.1-hiera-large"
# model_name = "facebook/sam2.1-hiera-tiny"

# https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L64
mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
    model_name,
    points_per_side=64,  # pts per length/width of image
    points_per_batch=128,  # higher = faster but use more gpu
    crop_n_layers=1,  # if >0 run again on crops of the img
)

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# Build VOS-optimized predictor
vid_predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=False
)

def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.7 + (i % 3) * 0.1  # Slight variation in saturation
        value = 0.8 + (i % 2) * 0.2  # Slight variation in value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return np.array(colors)


def create_segm_img(masks):
    # masks len num_objects, each one is bool (H, W)
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 3))

    color_list = generate_distinct_colors(len(masks))
    for i, mask in enumerate(masks):
        img[mask] = color_list[i]
    return img
    

def predict_segm_video(video_filepath):
    # read in images from input video
    with smart_open.open(video_filepath, "rb") as f:
        reader = decord.VideoReader(f, num_threads=8)
        fps = reader.get_avg_fps()
        
        idxs = range(len(reader))
        images = reader.get_batch(idxs)  # (T, H, W, 3), uint8, 0 to 255
        # down sample the resolution half
        if images.dtype==torch.uint8:
            images = images.numpy()
        else:
            images = images.asnumpy()
        # new_images = [] 
        # for i in range(len(images)):
        #     new_images.append(cv2.resize(images[i], (images[i].shape[2]//2, images[i].shape[1]//2)))
        # images = new_images

    # initialize inference state
    inference_state = vid_predictor.init_state(video_path=video_filepath)

    # add some points for what to track throughout rest of the video
    initial_frame_masks = mask_generator.generate(images[0])
    for i, object_frame_mask in enumerate(initial_frame_masks):
        _, object_ids, mask_logits = vid_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i,
            points=object_frame_mask["point_coords"],
            labels=np.ones(1),
        )

    # track the above points throughout the video
    segm_imgs = []
    for frame_idx, object_ids, mask_logits in vid_predictor.propagate_in_video(inference_state):
        frame = images[frame_idx]
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        segm = create_segm_img(masks)
        segm_imgs.append((segm * 255).astype("uint8"))  
    return segm_imgs, fps
        

# @click.command()
# @click.option("--manifest-path", type=click.Path(exists=True))
# def main(manifest_path: str):
#     with open(os.path.expanduser(manifest_path), "r") as f:
#         for line in f:
#             rgb_filepath = f"{line.strip()}/image.mp4"
#             output_filepath = rgb_filepath.replace("image.mp4", "video_segm.mp4")
            
#             print(f"\nProcessing {rgb_filepath}...")
#             segm_imgs, fps = predict_segm_video(rgb_filepath)
            
#             clip = ImageSequenceClip(segm_imgs, fps=fps)
#             clip.write_videofile(output_filepath, codec='libx264')


def write_video_with_opencv(segm_imgs, output_filepath, fps):
    """Write video using OpenCV instead of moviepy"""
    if not segm_imgs:
        print("No images to write")
        return
    
    # Get dimensions from first image
    height, width = segm_imgs[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    
    for img in segm_imgs:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    # Release everything
    out.release()
    print(f"Video saved to {output_filepath}")


def main(manifest_path: str = None):
    rgb_filepath = "demo/data/gallery/0.mp4"
    output_filepath = rgb_filepath.replace("0.mp4", "video_segm.mp4")
    
    print(f"\nProcessing {rgb_filepath}...")
    segm_imgs, fps = predict_segm_video(rgb_filepath)
    
    write_video_with_opencv(segm_imgs, output_filepath, fps)

if __name__ == "__main__":
    """Supplement existing rgb data (.../image.mp4) with corresponding segm images (.../video_segm.mp4).
    
    $ pip install transformers huggingface_hub==0.25.0
    $ pip install hydra-core iopath
    
    # note: not installing everything from them bc it changes all our versions
    $ git clone https://github.com/facebookresearch/sam2.git 
    $ export PYTHONPATH=~/sam2:$PYTHONPATH
    
    $ python3 create_video_segm_data.py --manifest-path ~/far_pi/manifest.txt
    """
    main()