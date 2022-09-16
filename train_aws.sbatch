#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%J_stdout.txt
#SBATCH --error=%x_%J_stderr.txt
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=brian.hempel@noaa.gov
#SBATCH --mail-type=ALL
##SBATCH --mem=100G

# USAGE: DATA_ROOT=$(pwd) SREF_OR_HREF=href FORECAST_HOUR_START=2 FORECAST_HOUR_STOP=13 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 EVENT_TYPES=; DATA_ROOT=$DATA_ROOT SREF_OR_HREF=$SREF_OR_HREF FORECAST_HOUR_START=$FORECAST_HOUR_START FORECAST_HOUR_STOP=$FORECAST_HOUR_STOP DATA_SUBSET_RATIO=$DATA_SUBSET_RATIO NEAR_STORM_RATIO=$NEAR_STORM_RATIO EVENT_TYPES=$EVENT_TYPES sbatch --job-name=train_${SREF_OR_HREF}_f${FORECAST_HOUR_START}-${FORECAST_HOUR_STOP}_${DATA_SUBSET_RATIO}_${NEAR_STORM_RATIO}_${EVENT_TYPES} /scratch/nadocast_dev/train_aws.sbatch
# DATA_ROOT=$(pwd) SREF_OR_HREF=sref FORECAST_HOUR_START=2 FORECAST_HOUR_STOP=13 DATA_SUBSET_RATIO=0.3 NEAR_STORM_RATIO= EVENT_TYPES=sig_tornado,sig_wind,tornado,wind; DATA_ROOT=$DATA_ROOT SREF_OR_HREF=$SREF_OR_HREF FORECAST_HOUR_START=$FORECAST_HOUR_START FORECAST_HOUR_STOP=$FORECAST_HOUR_STOP DATA_SUBSET_RATIO=$DATA_SUBSET_RATIO NEAR_STORM_RATIO=$NEAR_STORM_RATIO EVENT_TYPES=$EVENT_TYPES sbatch --job-name=train_${SREF_OR_HREF}_f${FORECAST_HOUR_START}-${FORECAST_HOUR_STOP}_${DATA_SUBSET_RATIO}_${NEAR_STORM_RATIO}_${EVENT_TYPES} /scratch/nadocast_dev/train_aws.sbatch

export HOME=/home/brian.hempel
export PATH=$PATH:$HOME/.local/bin:$HOME/bin

export FORECAST_HOUR_RANGE=${FORECAST_HOUR_START}:${FORECAST_HOUR_STOP}
export JULIA_MPI_BINARY=system
export JULIA_MPI_PATH=/opt/amazon/openmpi
export JULIA_MPI_LIBRARY=/opt/amazon/openmpi/lib64/libmpi
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MUST_LOAD_FROM_DISK=true
export DISTRIBUTED=true

cd $DATA_ROOT

pwd

env

cat /proc/cpuinfo

time mpirun --bind-to none julia --compiled-modules=no --project=/scratch/nadocast_dev /scratch/nadocast_dev/models/${SREF_OR_HREF}_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
