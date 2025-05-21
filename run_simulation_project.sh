#!/usr/bin/env bash
 
stage=$1
from_task1=$2          # 1: start from task1, 0: start from task2
task2_start_from=$3    # gp  |  rand 

path_data=./data/project_experiments/


# check arguments
if [ -z $1 ] || [ -z $2 ]; then 
    echo "Usage: bash run_simulation_sampling.sh <start_from_task1> <task2_start_from>"
    echo "<stage-number>: 1, run 10-D sampling experiments;"
    echo "<start_from_task1>: 0, skip task1 and run task 2;"
    echo "<start_from_task1>: 1, run from task1;"
    echo "<start_from_task1>: 2, run task1 only;"
    echo "<task2_start_from>: gp or rand, run task2 from gp or rand in task1" && exit 0
fi


if [ $stage -eq 1 ]; then 
    echo "Simulation 1: transfer optimization from fixed source triple 10-D exponential function to sampled family functions"

    dim=10
    num_rep=100
    T1=17      # dim * 20

    num_task2=1
    T2=10

    for i in $(seq 1 $num_rep); do
        echo "Task 1: 10-D triple exponential function; Taks 2: sampled family function, starts from best point in $task2_start_from" 
        echo "Running $i-th round simulation"
    
        mkdir -p $path_data/Dimension-10_20T1/$i
        out_dir=$path_data/Dimension-10_20T1/$i
    
        job_name=project_sampled_${task2_start_from}_$i
        sbatch  --job-name=$job_name  ./main_sample_simulation.py  --dim $dim  --T1 $T1  --T2 $T2  --num_task2 $num_task2 \
                                        --task2_start_from $task2_start_from  --from_task1 $from_task1  --out_dir $out_dir
    
        echo "Submitted $i-th sampled simulation by Slurm" 
    done
fi

