far-set-env EKS
kubectl config use-context sky-us-east-2
sky api stop  # need to run this when you switch the env

# check the number of GPUs avaialable
# echo $((
# $(kubectl get nodes -l node.kubernetes.io/instance-type=p4d.24xlarge -o json | jq '[.items[].status.allocatable."nvidia.com/gpu" | tonumber] | add') -
# $(kubectl get pods -A -o json | jq '[.items[].spec.containers[]?.resources.limits."nvidia.com/gpu" | select(. != null) | tonumber] | add')
# ))

# launch jobs
FAR-skypilot-wrapper/bin/sky-ws42-launch.sh EKS:openpi -y -c $USER-groot_eval \
 --num-nodes 1 --gpus A100:1 --infra k8s/sky-us-east-2 \
--image-id 241533154612.dkr.ecr.us-east-1.amazonaws.com/dexmimicgen:high \
--env ENTRYPOINT="bash"