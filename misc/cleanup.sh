cd ~/sky_workdir/FAR-affordance/molmo/finetuned_checkpoints_action


while true; do  
    for dir in step*-unsharded; do
        if [[ -d "$dir" ]]; then
            step=$(echo "$dir" | sed 's/step\([0-9]*\)-unsharded/\1/')
            if [[ -n "$step" ]] && (( step % 10000 == 0 )); then
                echo "Uploading $dir (step $step) to S3..."
                aws s3 sync "$dir" "s3://far-research-internal/ishikasi/checkpoints/action_chunck_100/step${step}/"
                aws s3 ls "s3://far-research-internal/ishikasi/checkpoints/action_chunck_100/step${step}/"
            else
                echo "Deleting $dir (step $step)..."
                rm -rf "$dir"
            fi
        fi
    done
    sleep 3600
done