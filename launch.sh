cd /root/sky_workdir/FAR-affordance/molmo/
conda activate molmo
wandb login --host=https://far.wandb.io
PYTHONPATH=. torchrun --nproc-per-node 8 --nnodes=${SKYPILOT_NUM_NODES} --node_rank=${SKYPILOT_NODE_RANK} --rdzv_id=311 --rdzv_endpoint=${SKYPILOT_MASTER_ADDR}:29500 launch_scripts/train_affordance.py trajectory_3d checkpoints/Molmo-7B-O-0924 --save_folder=finetuned_checkpoints --save_overwrite --wandb.name=trajectory_3d --wandb.entity=fanyangr --wandb.project=affordance --cotrain $((SKYPILOT_NUM_NODES*8*2)) --global_batch_size 32 --device_eval_batch_size 1 