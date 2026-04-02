# far-set-env SKY
# kubectl config use-context sky-us-east-2
# launch jobs
sky EKS:affordance -y -c $USER-dev-eks-test \
 --num-nodes 1 --gpus A100:4 --infra k8s/sky-us-east-2 \
--image-id 241533154612.dkr.ecr.us-east-1.amazonaws.com/dexmimicgen:affordance \
--env ENTRYPOINT="bash"