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
  --exclude='datasets' \
  --exclude='output' \
  --exclude='visualizations' \
  fanyangr-groot:~/sky_workdir/far_pi/ FAR-GR00T/
  # fanyangr-egodex:~/sky_workdir/far_pi/ FAR-HierarchicalVLA-high/