far-set-env SKY

# launch jobs
sky EKS:affordance -y -c $USER-trajectory-eks-multi \
 --num-nodes 4 --gpus A100:8 --infra k8s/sky-us-east-2 \
--image-id 241533154612.dkr.ecr.us-east-1.amazonaws.com/dexmimicgen:affordance \
--env ENTRYPOINT="/root/sky_workdir/FAR-affordance/launch.sh"