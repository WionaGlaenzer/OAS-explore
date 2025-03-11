# submit_and_wait.sh
#!/bin/bash
job_id=$(sbatch -A es_reddy -n 1 --cpus-per-task=4 --gpus=1 --gres=gpumem:20g \
          --time=24:00:00 --mem-per-cpu=50g --job-name="model_training" --ntasks=1 \
          --parsable \
          --wrap="$*")

echo "Submitted job $job_id"

# Wait for the job to complete
while true; do
    status=$(sacct -j $job_id --format=State --noheader | head -n 1 | tr -d ' ')
    if [[ "$status" == "COMPLETED" ]]; then
        echo "Job completed successfully"
        exit 0
    elif [[ "$status" == "FAILED" || "$status" == "CANCELLED" || "$status" == "TIMEOUT" ]]; then
        echo "Job failed with status: $status"
        exit 1
    fi
    sleep 60  # Check every minute
done