#!/usr/bin/env bash
 
stage=$1
from_task1=$2          # 1: start from task1, 0: start from task2
task2_start_from=$3    # gp  |  rand 

path_data=./data/sampling_experiments/


# check arguments
if [ -z $1 ] || [ -z $2 ]; then 
    echo "Usage: bash run_simulation_sampling.sh <start_from_task1> <task2_start_from>"
    echo "<stage-number>: 1, run 1D sampling experiments;"
    echo "<stage-number>: 2, run 2D sampling experiments;"
    echo "<start_from_task1>: 0, skip task1 and run task 2;"
    echo "<start_from_task1>: 1, run from task1;"
    echo "<start_from_task1>: 2, run task1 only;"
    echo "<task2_start_from>: gp or rand, run task2 from gp or rand in task1" && exit 0
fi


if [ $stage -eq 1 ]; then 
    echo "Simulation 1: transfer optimization from fixed source triple 1-D exponential function to sampled family functions"

    dim=1
    num_rep=100
    T1=20

    num_task2=2
    T2=20

    for i in $(seq 1 $num_rep); do
        echo "Task 1: triple exponential function; Taks 2: sampled family function, starts from best point in $task2_start_from" 
        echo "Running $i-th round simulation"
    
        mkdir -p $path_data/Dimension-1/$i
        out_dir=$path_data/Dimension-1/$i
    
        job_name=simulation_sampled_${task2_start_from}_$i
        sbatch  --job-name=$job_name  ./main_sample_simulation.py  --dim $dim --T1 $T1  --T2 $T2  --num_task2 $num_task2 \
                                        --task2_start_from $task2_start_from  --from_task1 $from_task1  --out_dir $out_dir
    
        echo "Submitted $i-th sampled simulation by Slurm" 
    done
fi 


if [ $stage -eq 2 ]; then 
    echo "Simulation 2: transfer optimization from fixed source triple 2-D exponential function to sampled family functions"

    dim=2
    num_rep=100
    T1=40

    num_task2=2
    T2=40

    for i in $(seq 1 $num_rep); do
        echo "Task 1: triple exponential function; Taks 2: sampled family function, starts from best point in $task2_start_from" 
        echo "Running $i-th round simulation"
    
        mkdir -p $path_data/Dimension-2/$i
        out_dir=$path_data/Dimension-2/$i
    
        job_name=simulation_sampled_${task2_start_from}_$i
        sbatch  --job-name=$job_name  ./main_sample_simulation.py  --dim $dim  --T1 $T1  --T2 $T2  --num_task2 $num_task2 \
                                        --task2_start_from $task2_start_from  --from_task1 $from_task1  --out_dir $out_dir
    
        echo "Submitted $i-th sampled simulation by Slurm" 
    done
fi

