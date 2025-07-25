set -ex
bazel build covariant/models/llm/scripts:nccl_test
torchrun \
    --nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE:-`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`} \
    --node_rank=${SKYPILOT_NODE_RANK:-0} \
    --nnodes=${SKYPILOT_NUM_NODES:-1} \
    --master_addr=${SKYPILOT_MASTER_ADDR:-localhost} \
    --master_port=9009 \
    /root/sky_workdir/FAR-affordance/nccl_test.py \
    $@
