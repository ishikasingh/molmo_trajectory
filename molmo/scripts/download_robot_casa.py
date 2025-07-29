from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

repo_id = "ishika/robocasa_gr1_tabletop_tasks_fingertips_corrected"
ds_meta = LeRobotDatasetMetadata(repo_id, revision="main")
# hub_api = HfApi()
# # You'll need to determine the correct version from the dataset's info.json
# # For now, let's try a common version like "v1.0.0"
# hub_api.create_tag(repo_id, tag="v1.0.0", repo_type="dataset")
dataset = LeRobotDataset(repo_id)
print(dataset[0])