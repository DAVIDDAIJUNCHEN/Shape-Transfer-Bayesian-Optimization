#!/usr/bin/env bash

stage=$1   # 0: run both EXP and BR; 1: run EXP only; 2: run BR only

# check arguments
if [ -z $1 ]; then 
    echo "Usage: bash run_simulation.sh <stage-number>" 
    echo "<stage-number>: 0, run all simulation;"
    echo "<stage-number>: 1, run EXP only;"
    echo "<stage-number>: 2, run BR only." && exit 0
fi


if [ $stage -eq 0 ] || [ $stage -eq 1 ]; then 
    echo "Simulation 1: Exponential target function"
    # Taks 1: mu_1, theta; Taks 2: mu_2, theta
    Thetas="1"
    mu_1="0.0_0.0"
    mu_2="1.0_1.0"

    T1=15
    T2=4

    num_rep=1

    for i in $(seq 1 $num_rep); do
        for theta in $Thetas; do
            python ./main_simulation.py  --T1 $T1  --T2 $T2  --type EXP  --mu1 $mu_1  --mu2 $mu_2  --theta $theta 

        done
    done 
fi

if [ $stage -eq 0 ] || [ $stage -eq 2 ]; then
    echo "Simulation 2: Branni function (task1), Modified Branni function (task2)"
    # Task 1: Branni; Task 2: Modified Branni
    T1=15
    T2=4

    num_rep=1

    for i in $(seq 1 $num_rep); do 
        python ./main_simulation.py --T1 $T1  --T2 $T2 --type BR 

    done
fi
