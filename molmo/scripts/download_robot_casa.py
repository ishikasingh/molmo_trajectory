from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

repo_id = "ishika/robocasa_gr1_tabletop_tasks_fingertips"
ds_meta = LeRobotDatasetMetadata(repo_id)
dataset = LeRobotDataset(repo_id)