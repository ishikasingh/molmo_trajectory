# far-set-env SKY
# launch jobs
sky LLKUB:affordance -y -c $USER-dev-lambda \
--num-nodes 1 --gpus H200:8 --image-id 241533154612.dkr.ecr.us-east-1.amazonaws.com/dexmimicgen:affordance\
 --env ENTRYPOINT="/root/sky_workdir/FAR-affordance/launch.sh" --retry-until-up