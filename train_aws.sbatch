#!/bin/bash
#SBATCH --partition=largejobs
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --nodes=7
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --output=train_${SREF_OR_HREF}_f${FORECAST_HOUR_START}-${FORECAST_HOUR_STOP}_${DATA_SUBSET_RATIO}_${NEAR_STORM_RATIO}_${EVENT_TYPES}_%J_stdout.txt
#SBATCH --error=train_${SREF_OR_HREF}_f${FORECAST_HOUR_START}-${FORECAST_HOUR_STOP}_${DATA_SUBSET_RATIO}_${NEAR_STORM_RATIO}_${EVENT_TYPES}_%J_stderr.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=train_${SREF_OR_HREF}_f${FORECAST_HOUR_START}-${FORECAST_HOUR_STOP}_${DATA_SUBSET_RATIO}_${NEAR_STORM_RATIO}_${EVENT_TYPES}
#SBATCH --mail-user=brian.hempel@noaa.gov
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/

# USAGE: SREF_OR_HREF=sref FORECAST_HOUR_START=12 FORECAST_HOUR_STOP=23 DATA_SUBSET_RATIO=0.3 NEAR_STORM_RATIO=0.4 EVENT_TYPES= sbatch train_aws.sbatch

export HOME=/home/brian.hempel
export PATH=$PATH:$HOME/.local/bin:$HOME/bin

export FORECAST_HOUR_RANGE=${FORECAST_HOUR_START}:${FORECAST_HOUR_STOP}
export JULIA_MPI_BINARY=system
export JULIA_MPI_PATH=/opt/amazon/openmpi
export JULIA_MPI_LIBRARY=/opt/amazon/openmpi/lib64/libmpi
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MUST_LOAD_FROM_DISK=true
export DISTRIBUTED=true
# env
time mpirun --bind-to none julia --compiled-modules=no --project=/scratch/nadocast_dev /scratch/nadocast_dev/models/${SREF_OR_HREF}_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
