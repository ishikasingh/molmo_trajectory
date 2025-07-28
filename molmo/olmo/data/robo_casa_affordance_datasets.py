import json
import numpy as np
import datasets
from PIL import Image, ImageDraw, ImageFont
from olmo.data.dataset import Dataset
from pathlib import Path
from pprint import pprint
import time
import random
# import matplotlib.pyplot as plt
import cv2
import warnings
import torch

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*", category=UserWarning)

class RobotCasaHandPositioningDataset(Dataset):
    def __init__(self):
        """
        data_path: Path to your dataset
        split: "train", "validation", or "test"
        max_retries: Maximum number of retry attempts for loading dataset
        retry_delay: Base delay between retries (will be randomized with jitter)
        """
        self.repo_id = "ishika/robocasa_gr1_tabletop_tasks_fingertips"
        
        # Load metadata with retry logic
        self.metadata = LeRobotDatasetMetadata(self.repo_id)
        # Load dataset with retry logic
        self.data = LeRobotDataset(self.repo_id, video_backend="pyav")
    
    def __len__(self):
        return len(self.data)
    
    def get(self, item, rng):
        example = self.data[item]
        image_tensor = example["observation.images.egoview"]  # torch [3, 256, 256]
        # Convert torch tensor to PIL Image
        image = Image.fromarray(
            (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        # Get image dimensions for normalization
        img_h, img_w = image.size[1], image.size[0]  # PIL Image size is (width, height)

        hand_position_2d = example["observation.condition_2D"].reshape(12, 2)  # torch [12, 2]

        # swap the order of the fingers to match the order in the affordance_datsets.py
        left_fingers = hand_position_2d[5:10]
        right_fingers = hand_position_2d[0:5]
        left_wrist = hand_position_2d[10:11]
        right_wrist = hand_position_2d[11:12]
        # note: the order for each keypoint is:
        # [left hand, left thumb, left index, left middle, left ring, left pinky, right hand, right thumb, right index, right middle, right ring, right pinky]
        hand_positions = np.concatenate([left_wrist,left_fingers, right_wrist, right_fingers], axis=0)
        assert hand_positions.shape == (12, 2)
        
        # Normalize the hand positions to percentage coordinates (vectorized)
        hand_positions = (hand_position_2d.numpy() / [img_w, img_h]) * 100.0

        language_instruction = example["task"]
        language_instruction = self.language_mapping(language_instruction)
        
        return {
            "image": image,
            "message_list": [
                {
                    "label": language_instruction,
                    "points": hand_positions,
                    "point_scale": 100,  # Our coordinates are already in percentage (0-100)
                    "style": "affordance"
                }
            ],
            "metadata": {
                # "image": image,  # Add this line - put image in metadata too
                # "image_path": safe_get(example, "image_path", ""),
                # "hand_data": example["hand_positions"],  # Keep original for debugging
                # "video_id": safe_get(metadata, "video_id", ""),
                # "frame_idx": safe_get(metadata, "frame_idx", 0),
                # "image_size": safe_get(metadata, "image_size", [1920, 1080])
            }
        }
    
    # def visualize_hand_positions(self, item, save_path=None):
    #     """
    #     Visualize hand positions on the image for a given item.
        
    #     Args:
    #         item: Index of the item to visualize
    #         save_path: Optional path to save the visualization
    #     """
    #     data = self.get(item, None)
    #     image = data["image"]
    #     hand_positions = data["message_list"][0]["points"]
    #     instruction = data["message_list"][0]["label"]
        
    #     # Convert percentage coordinates back to pixel coordinates
    #     img_w, img_h = image.size
    #     pixel_positions = (hand_positions / 100.0) * np.array([img_w, img_h])
        
    #     # Create a copy of the image for drawing
    #     vis_image = image.copy()
    #     draw = ImageDraw.Draw(vis_image)
        
    #     # Define colors for different hand parts (12 points total)
    #     colors = [
    #         (255, 0, 0),    # Red - thumb tip
    #         (255, 100, 0),  # Orange - thumb IP
    #         (255, 200, 0),  # Yellow - thumb MCP
    #         (0, 255, 0),    # Green - index tip
    #         (0, 255, 100),  # Light green - index PIP
    #         (0, 255, 200),  # Lighter green - index MCP
    #         (0, 0, 255),    # Blue - middle tip
    #         (100, 0, 255),  # Purple - middle PIP
    #         (200, 0, 255),  # Magenta - middle MCP
    #         (255, 0, 255),  # Pink - ring tip
    #         (255, 100, 255), # Light pink - ring PIP
    #         (255, 200, 255)  # Lighter pink - ring MCP
    #     ]
        
    #     # Draw hand position points
    #     point_radius = 3
    #     for i, (x, y) in enumerate(pixel_positions):
    #         color = colors[i % len(colors)]
    #         # Draw filled circle
    #         draw.ellipse([x - point_radius, y - point_radius, 
    #                      x + point_radius, y + point_radius], 
    #                     fill=color, outline=(255, 255, 255))
    #         # Add point number
    #         draw.text((x + 5, y - 5), str(i), fill=(255, 255, 255))
        
    #     # Add instruction text
    #     try:
    #         # Try to use a default font
    #         font = ImageFont.load_default()
    #     except:
    #         font = None
        
    #     # Add text at the top of the image
    #     text_y = 10
    #     draw.text((10, text_y), f"Instruction: {instruction}", 
    #              fill=(255, 255, 255), font=font)
    #     draw.text((10, text_y + 20), f"Hand positions (12 points):", 
    #              fill=(255, 255, 255), font=font)
        
    #     # Print coordinate information
    #     print(f"\n=== Item {item} ===")
    #     print(f"Instruction: {instruction}")
    #     print(f"Image size: {img_w} x {img_h}")
    #     print("Hand positions (pixel coordinates):")
    #     for i, (x, y) in enumerate(pixel_positions):
    #         print(f"  Point {i}: ({x:.1f}, {y:.1f})")
        
    #     if save_path:
    #         vis_image.save(save_path)
    #         print(f"Visualization saved to: {save_path}")
        
    #     return vis_image

    # def visualize_hand_positions_video(self, start_idx, num_frames=30, fps=10, save_path="hand_positions_video.mp4"):
    #     """
    #     Create a video visualization of hand positions across consecutive frames.
        
    #     Args:
    #         start_idx: Starting index for the sequence
    #         num_frames: Number of consecutive frames to include
    #         fps: Frames per second for the video
    #         save_path: Path to save the video file
    #     """
    #     # Get video dimensions from first frame
    #     first_data = self.get(start_idx, None)
    #     first_image = first_data["image"]
    #     img_w, img_h = first_image.size
        
    #     # Initialize video writer
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(save_path, fourcc, fps, (img_w, img_h))
        
    #     print(f"Creating video with {num_frames} frames starting from index {start_idx}")
    #     print(f"Video settings: {img_w}x{img_h}, {fps} FPS")
        
    #     for i in range(num_frames):
    #         frame_idx = start_idx + i
    #         if frame_idx >= len(self.data):
    #             print(f"Reached end of dataset at frame {i}")
    #             break
                
    #         print(f"Processing frame {i+1}/{num_frames} (dataset index {frame_idx})")
            
    #         # Get data for this frame
    #         data = self.get(frame_idx, None)
    #         image = data["image"]
    #         hand_positions = data["message_list"][0]["points"]
    #         instruction = data["message_list"][0]["label"]
            
    #         # Convert percentage coordinates back to pixel coordinates
    #         pixel_positions = (hand_positions / 100.0) * np.array([img_w, img_h])
            
    #         # Create a copy of the image for drawing
    #         vis_image = image.copy()
    #         draw = ImageDraw.Draw(vis_image)
            
    #         # Define colors for different hand parts (12 points total)
    #         colors = [
    #             (255, 0, 0),    # Red - thumb tip
    #             (255, 100, 0),  # Orange - thumb IP
    #             (255, 200, 0),  # Yellow - thumb MCP
    #             (0, 255, 0),    # Green - index tip
    #             (0, 255, 100),  # Light green - index PIP
    #             (0, 255, 200),  # Lighter green - index MCP
    #             (0, 0, 255),    # Blue - middle tip
    #             (100, 0, 255),  # Purple - middle PIP
    #             (200, 0, 255),  # Magenta - middle MCP
    #             (255, 0, 255),  # Pink - ring tip
    #             (255, 100, 255), # Light pink - ring PIP
    #             (255, 200, 255)  # Lighter pink - ring MCP
    #         ]
            
    #         # Draw hand position points
    #         point_radius = 3
    #         for j, (x, y) in enumerate(pixel_positions):
    #             color = colors[j % len(colors)]
    #             # Draw filled circle
    #             draw.ellipse([x - point_radius, y - point_radius, 
    #                          x + point_radius, y + point_radius], 
    #                         fill=color, outline=(255, 255, 255))
    #             # Add point number
    #             draw.text((x + 5, y - 5), str(j), fill=(255, 255, 255))
            
    #         # Add instruction text and frame info
    #         try:
    #             font = ImageFont.load_default()
    #         except:
    #             font = None
            
    #         # Add text at the top of the image
    #         text_y = 10
    #         draw.text((10, text_y), f"Frame: {frame_idx} | Instruction: {instruction}", 
    #                  fill=(255, 255, 255), font=font)
    #         draw.text((10, text_y + 20), f"Hand positions (12 points) - Frame {i+1}/{num_frames}", 
    #                  fill=(255, 255, 255), font=font)
            
    #         # Convert PIL image to OpenCV format (RGB to BGR)
    #         vis_array = np.array(vis_image)
    #         vis_bgr = cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR)
            
    #         # Write frame to video
    #         video_writer.write(vis_bgr)
        
    #     # Release video writer
    #     video_writer.release()
    #     print(f"Video saved to: {save_path}")
        
    #     return save_path

    def language_mapping(self, language_instruction):
        mapping_mapping = {
            "PosttrainPnPNovelFromPlacematToTieredshelfSplitA": "Pick up the object on the placemat and place it on the shelf",
            "PosttrainPnPNovelFromCuttingboardToBasketSplitA": "Pick up the object on the cutting board and place it in the basket",
            "PosttrainPnPNovelFromTrayToTieredshelfSplitA": "Pick up the object on the tray and place it on the tiered shelf",
            "PnPBottleToCabinetClose": "Pick up the bottle and place it in the cabinet",
            "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA": "Pick up the object on the cutting board and place it in the basket",
            "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA": "Pick up the object on the cutting board and place it in the cardboard box",
            "PosttrainPnPNovelFromPlateToPanSplitA": "Pick up the object on the plate and place it on the pan",
            "PosttrainPnPNovelFromPlacematToBasketSplitA": "Pick up the object on the placemat and place it in the basket",
            "PosttrainPnPNovelFromTrayToCardboardboxSplitA": "Pick up the object on the tray and place it in the cardboard box",
            "PosttrainPnPNovelFromCuttingboardToPotSplitA": "Pick up the object on the cutting board and place it in the pot",
            "PnPWineToCabinetClose": "Pick up the wine and place it in the cabinet",
            "PosttrainPnPNovelFromTrayToPotSplitA": "Pick up the object on the tray and place it in the pot",
            "PosttrainPnPNovelFromPlateToBowlSplitA": "Pick up the object on the plate and place it in the bowl",
            "PosttrainPnPNovelFromTrayToPlateSplitA": "Pick up the object on the tray and place it on the plate",
            "PnPMilkToMicrowaveClose": "Pick up the milk and place it in the microwave",
            "PnPPotatoToMicrowaveClose": "Pick up the potato and place it in the microwave",
            "PosttrainPnPNovelFromCuttingboardToPanSplitA": "Pick up the object on the cutting board and place it on the pan",
            "PosttrainPnPNovelFromPlacematToPlateSplitA": "Pick up the object on the placemat and place it on the plate",
            "PnPCupToDrawerClose": "Pick up the cup and place it in the drawer",
            "PnPCanToDrawerClose": "Pick up the can and place it in the drawer"
        }
        return mapping_mapping[language_instruction]

if __name__ == "__main__":
    dataset = RobotCasaHandPositioningDataset()
    
    # Option 1: Create a video from consecutive frames
    print("Creating video from consecutive frames...")
    video_path = dataset.visualize_hand_positions_video(
        start_idx=0, 
        num_frames=1000,  # 30 consecutive frames
        fps=30,         # 10 frames per second
        save_path="hand_positions_sequence.mp4"
    )
    
    # Option 2: Save individual frames as images
    # print("\nSaving individual frames...")
    # dataset.visualize_hand_positions_sequence(
    #     start_idx=0,
    #     num_frames=10,  # 10 consecutive frames
    #     save_dir="frame_sequence"
    # )
    
    # # Option 3: Visualize a few individual examples (original functionality)
    # print("\nVisualizing individual examples...")
    # num_examples = 3
    # for i in range(min(num_examples, len(dataset))):
    #     print(f"\n{'='*50}")
    #     print(f"Visualizing example {i}")
    #     print(f"{'='*50}")
        
    #     # Create visualization
    #     vis_image = dataset.visualize_hand_positions(i, save_path=f"hand_positions_example_{i}.png")
        
    #     # Display the image (if matplotlib is available)
    #     try:
    #         plt.figure(figsize=(12, 8))
    #         plt.imshow(vis_image)
    #         plt.title(f"Hand Positions - Example {i}")
    #         plt.axis('off')
    #         plt.show()
    #     except Exception as e:
    #         print(f"Could not display image: {e}")
    #         print("Image saved as PNG file instead.")
    
    # print(f"\nAll visualizations complete!")
    # print(f"- Video: {video_path}")
    # print(f"- Individual frames: frame_sequence/")
    # print(f"- Individual examples: hand_positions_example_*.png")