rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='dataset' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='node_modules' \
  --exclude='third_party' \
  --exclude='wandb' \
  --exclude='checkpoints' \
  --exclude='hand_trajectory_visualizations' \
  fanyangr-egodex:~/sky_workdir/far_pi/ FAR-HierarchicalVLA-high/
  # fanyangr-egodex:~/sky_workdir/far_pi/ FAR-HierarchicalVLA-high/