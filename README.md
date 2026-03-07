## repo 
The code base is at https://code.amazon.com/packages/FAR-affordance/trees/mainline
## Setup sky pilot config files
At the FAR-skypilot-wrapper, switch to branch `dev/fanyangr/affordance`. The files I made changes are `affordance.yaml`(there are several affordance.yaml files, each under a different folder, corresponding to the specific server you want to launch this to.), and `config.json` file to inform sky where my personally defined yaml files are. If you want to keep using your original branch under FAR-skypilot-wrapper, you can copy those files over. 
## Launch compute nodes on clusters
There are three scripts: `launch_lambda_affordance.sh`(launch a H200 node), `launch_eks_affordance.sh` (launch a A100 node), `launch_eks_affordance_multi.sh` (launch several A100 nodes and use them for training). Those scripts will launch nodes, download dataset and pretrained VLM, and set environment variables. You should commit your changes before you wrong. PS: when I ran my code, those scripts are under the lab42/src folder, I am not sure if changing it here will still work. 
## Launch the training code
On the cluster node, first acticate the conda environment `conda activate molmo`. Then `cd ./sky_workdir/FAR-affordance/molmo/`. after that you should be able to launch the training script. The script will be the same for single-node and multi-node training. An example would be 
```
PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py trajectory_3d_human_robot_fm pretrained_checkpoints/ --save_folder=finetuned_checkpoints --save_overwrite --wandb.name=seperate_0051 --wandb.entity=fanyangr --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*16)) --device_train_batch_size 16 --device_eval_batch_size 1 --seq_len 900 --action_horizon 30 --finetune --action_expert_mode separate_human_robot --robot_use_joint_action

PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py trajectory_3d_human_robot_fm checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints --save_overwrite --wandb.name=human_robot_3d_fm_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*2)) --device_train_batch_size 2 --device_eval_batch_size 1 --seq_len 500 --action_horizon 30 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm

PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py trajectory_3d_human_robot_action_fm checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_action --save_overwrite --wandb.name=human_robot_3d_fm_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*16)) --device_train_batch_size 16 --device_eval_batch_size 1 --seq_len 500 --action_horizon 100 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm

# action only
PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py robot_action_direct checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_action --save_overwrite --wandb.name=human_robot_3d_fm_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*48)) --device_train_batch_size 48 --device_eval_batch_size 1 --seq_len 500 --action_horizon 100 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm

PYTHONPATH=. torchrun --nproc-per-node 4 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py robot_action_direct checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_action_actpad_ds10 --save_overwrite --wandb.name=human_robot_3d_fm_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*48)) --device_train_batch_size 48 --device_eval_batch_size 1 --seq_len 500 --action_horizon 100 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm --pad_action_chunk

# sequential
PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py trajectory_3d_human_robot_fm checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_action --save_overwrite --wandb.name=human_robot_3d_fm_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*8*48)) --device_train_batch_size 48 --device_eval_batch_size 1 --seq_len 500 --action_horizon 100 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm --action_expert_mode sequential

# aloha trosson
PYTHONPATH=. torchrun --nproc-per-node 6 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py git stat checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_3d_traj_trossen --save_overwrite --wandb.name=human_trosson_3d_direct_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*6*12)) --device_train_batch_size 12 --device_eval_batch_size 1 --seq_len 500 --action_horizon 30 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm --robot_trajectory_dim=6


PYTHONPATH=. torchrun --nproc-per-node 6 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py 
 checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_3d_traj_trossen_only_train --save_overwrite --wandb.name=trosson_3d_direct_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*6*12)) --device_train_batch_size 12 --device_eval_batch_size 1 --seq_len 500 --action_horizon 30 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm

<!-- trossen_3d_direct -->

export num_gpus=$(nvidia-smi -L | wc -l)
PYTHONPATH=. torchrun --nproc-per-node $num_gpus --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py  trajectory_3d_trossen_direct checkpoints/Molmo-7B-D-0924 --save_folder=finetuned_checkpoints_3d_traj_trossen_human_labelled_train --save_overwrite --wandb.name=trosson_human_3d_direct_labelled_0065 --wandb.entity=ishikasi --wandb.project=affordance --global_batch_size $((SKYPILOT_NUM_NODES*$num_gpus*12)) --device_train_batch_size 12 --device_eval_batch_size 1 --seq_len 500 --action_horizon 30 --max_crops 2 --flow_matching_prediction_type=x0 --freeze_vlm


```
note that the global batch size `--global_batch_size $((SKYPILOT_NUM_NODES*8*16))` should match `--device_train_batch_size 16`.
## Important arguments:
1. max_crops: choosing how many crops to use for image tokenization, more crops provide more details, but in practice I found that does not really affect the performance. I use 2 in practice and using fewer crops will significantly save memories and speed up training
2. seq_len: the memory is linear to the seq_len. As long as it is more than the maximum seq_len you need during training, you should be good. In practice when max_crops = 2, I use seq_len = 600
3. freeze_vlm: this freezes the vlm backbone and will significantly speed up the training
4. action_horizon: the action chunking horizon to predict (how many steps in the future to predict), default to be 30.
5. Since my code supports both fingertip trajectory prediction and joint action prediction, there are arguments `human_action_dim` `robot_action_dim`. `robot_trajectory_dim`.We also have `human_proprio_dim`, `robot_proprio_dim`. Currently it assumes the fingertip positions is also the proprioception of the system (I don't use robot joint states as proprioception as I think fingertip positions can share across different embodiments). 
6. `action_expert_mode`: This defines three different architecture. `disabled` should never be used, it is left for text-based output and i never used it. `shared` is the Pi-0 architecture where robot and human data shares the same action expert. `separate_human_robot` is similar to PI-0 but uses different action experts for different embodiments. `sequential` is concatenating two action experts together: first predicts fingertip positions, and the second predicts robot joint angles. 
7. flow_matching_prediction_type: velocity is the classic flow matching formulation, x0 is the x0-prediction. x0 prediction has a much faster convergence speed in practice.

8. `finetune`, `cotrain`, `debug`, `slow_warmup`. setup a specific hyper parameter for training. You don't need to worry about those and you can set your parameters. 
9. `use_transitions` is deprecated
## Data loading
When introducing a new dataset, you will want to compute the normalization stats of the dataset using `compute_trajectory_stats.py`. This will give you mean and vairance. Note that in `olmo\data\__init__.py`, we specifies how the data is loaded. The trajectory stats file are something that I precomputed and saved on s3 (you can recompute it again in case there is a bug). `trajectory_representation` can be either `delta`(essentially velocity prediction) or `absolute` (classic way of predicting the trajectory itself). It is default to delta prediction. However, for robocasa dataset and predicting robot joint actions, `delta` prediction is not defined as we do not know the robot joint states (the first state), which makes it impossible to compute the deltas. `frame_downsampling_ratio` specifies that we only sample every `n` frame from the dataset, as using every single frame is not data efficient.  

## Evaluation
In practice, I will launch another node with only one gpu just for evaluation.
### evaluate the fingertip position 
you will use `eval_trajectory_flow_matching.py`. The defaults values are ready to use. You need to specify the checkpoint you want to evaluate. `split` specifies whether you want to evaluate on training or test set. `dataset` specifies which dataset you want to evaluate on. 

### evaluate robocasa
Please refer to `eval_close_loop.py`

```
aws s3 sync s3://far-research-internal/ishikasi/checkpoints/action_chunck_100/ --region us-east-1

 aws s3 cp s3://far-research-internal/ishikasi/dataset/aloha_data_v1/trossen_ee_world.hdf5 ./ --region us-east-1
 export

 git clone https://huggingface.co/datasets/ishika/aloha_play_dataset_part_3_with_fk_full_split

 aws s3 cp s3://far-research-internal/ishikasi/datasets/trossen/trossen_ee_world.hdf5 aloha_play_dataset_part_3_with_fk_full_split/  --region us-east-1


  ulimit -n 4096 && aws s3 sync s3://far-research-internal/ishikasi/checkpoints/trossen_3d_traj_labelled/step120000-unsharded/ finetuned_checkpoints_3d_traj_trossen_only_labelled_train/step120000-unsharded/  --region us-east-1
  ulimit -n 4096 && aws s3 sync s3://far-research-internal/ishikasi/checkpoints/trossen_human_3d_traj_labelled/step75000-unsharded/ finetuned_checkpoints_3d_traj_trossen_human_labelled_train/step75000-unsharded/ --region us-east-1


python launch_scripts/eval_closed_loop.py finetuned_checkpoints_action_hz100/ --dataset_path /root/sky_workdir/dataset/robocasa_og/ --output_dir closed_loop_eval_results_hz50 --num_tasks 20 --steps_per_chunk 50 --action_horizon 100

```
