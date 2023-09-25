#!/usr/bin/env bash

stage=$1       # 0: run both EXP and BR; 1: run EXP only; 2: run BR only
from_task1=$2  # 1: start from task1, 0: start from task2

path_data=data/


# check arguments
if [ -z $1 ] || [ -z $2 ]; then 
    echo "Usage: bash run_simulation.sh <stage-number> <start_from_task1>" 
    echo "<stage-number>: 0, run all simulation;"
    echo "<stage-number>: 1, run EXP only;"
    echo "<stage-number>: 2, run Branin only;"
    echo "<start_from_task1>: 1, run from task1;"
    echo "<start_from_task1>: 0, skip task1 and run task 2." && exit 0
fi


if [ $stage -eq 0 ] || [ $stage -eq 1 ]; then 
    echo "Simulation 1: Exponential target function"

    Thetas="1"
    mu_1="0.0_0.0"
    mu_2="1.0_1.0"

    T1=15
    T2=5

    num_rep=1

    for theta in $Thetas; do
        echo "Taks 1: mean=$mu_1, theta=$theta; Taks 2: mean=$mu_2, theta=$theta"
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            python ./main_simulation.py  --T1 $T1  --T2 $T2  --type EXP  --mu1 $mu_1  --mu2 $mu_2  --theta $theta --from_task1 $from_task1
            mkdir -p $path_data/EXP_$theta/$i
            mv $path_data/simExp*tsv   $path_data/EXP_$theta/$i/
        done
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 2 ]; then
    echo "Simulation 2: Branin function (task1), Modified Branni function (task2)"
    T1=15
    T2=5

    num_rep=1

    for i in $(seq 1 $num_rep); do 
        echo "Running $i-th simulation"
        python ./main_simulation.py --T1 $T1  --T2 $T2 --type BR --from_task1 $from_task1
        mkdir -p $path_data/Branin/$i
        mv $path_data/simBr*tsv   $path_data/Branin/$i/
    done
fi
