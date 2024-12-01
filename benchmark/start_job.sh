#!/bin/bash

NODES=$1
GPUS_PER_NODE=$2
FILENAME=$3

# remove the extension from the filename
PROGRAM_NAME=$(echo $FILENAME | cut -f 1 -d '.')

# Get the number of GPUs
GPUS=$(($NODES * $GPUS_PER_NODE))

# Create a temporary job script
JOB_SCRIPT=$(mktemp /tmp/job.XXXXXX.sh)

# Write the sbatch script to the temporary file
cat <<EOL > $JOB_SCRIPT
#!/bin/bash
#SBATCH --nodes=$NODES
#SBATCH --gpus=$GPUS
#SBATCH --time=00:10:00
#SBATCH --account=uo0780
#SBATCH --partition=gpu
#SBATCH --job-name=$PROGRAM_NAME
#SBATCH --output=log/${PROGRAM_NAME}_%j.out

# Load necessary modules
source activate fridom

# Run the Python script
srun -l -n $GPUS python $FILENAME
EOL

# Submit the job with sbatch
sbatch $JOB_SCRIPT
