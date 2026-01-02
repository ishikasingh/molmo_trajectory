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
  --exclude='assets' \
  --exclude='hand_trajectory_visualizations' \
  FAR-HierarchicalVLA-high/ fanyangr-egodex:~/sky_workdir/far_pi/

# Alternative destination (commented out):
# FAR-HierarchicalVLA-high/ fanyangr-eval:~/sky_workdir/far_pi/
# FAR-HierarchicalVLA-high/ fanyangr-egodex:~/sky_workdir/far_pi/
