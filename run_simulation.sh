#!/usr/bin/env bash

stage=$1       # 0: run both EXP and BR; 1: run EXP only; 2: run BR only
from_task1=$2  # 1: start from task1, 0: start from task2
task2_start_from=$3

path_data=data/


# check arguments
if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then 
    echo "Usage: bash run_simulation.sh <stage-number> <start_from_task1>" 
    echo "<stage-number>: 0, run all types of simulations;"
    echo "<stage-number>: 1, run EXP only;"
    echo "<stage-number>: 2, run Branin only;"
    echo "<stage-number>: 3, run Needle only;"
    echo "<stage-number>: 4, run Mono2Needle only;"
    echo "<stage-number>: 5, run Mono2Double only;"
    echo "<stage-number>: 6, run Double2Double only;"
    echo "<stage-number>: 7, run Triple2Double only;"
    echo "<stage-number>: 8, run Double2Triple only;"   
    echo "<stage-number>: 9, run Triple2Triple 2D only;" 
    echo "<start_from_task1>: 0, skip task1 and run task 2;"
    echo "<start_from_task1>: 1, run from task1;" 
    echo "<start_from_task1>: 2, run task1 only"
    echo "<task2_start_from>: gp or rand, run task2 from gp or rand in task1" && exit 0
fi


if [ $stage -eq 0 ] || [ $stage -eq 1 ]; then 
    echo "Simulation 1: Transfer Bayesian Optimization on Exponential target function"

    Thetas="0.5"
    mu_1="0_0"
    mu_2="0.1_0.1"

    T1=20
    T2=20

    num_rep=20

    for theta in $Thetas; do
        echo "Task 1: mean=$mu_1, theta=$theta; Taks 2: mean=$mu_2, theta=$theta, starts from best point in $task2_start_from" 
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            mkdir -p $path_data/EXP_mu2_${mu_2}_theta_${theta}_sample/$i
            out_dir=$path_data/EXP_mu2_${mu_2}_theta_${theta}_sample/$i

            job_name=EXP_mu2_${mu_2}_theta_${theta}_${task2_start_from}_$i
            sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                            --type EXP  --mu1 $mu_1  --mu2 $mu_2  --theta $theta  --from_task1 $from_task1
            echo "Submitted $i-th EXP simulation by Slurm" 
        done
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 2 ]; then
    echo "Simulation 2: Branin function (task1), Modified Branni function (task2) starts from best point in $task2_start_from"
    T1=20
    T2=20

    num_rep=20

    for i in $(seq 1 $num_rep); do 
        echo "Running $i-th simulation"
        mkdir -p $path_data/Branin/$i
        out_dir=$path_data/Branin/$i

        job_name=Branin_${task2_start_from}_$i
        sbatch --job-name=$job_name ./main_simulation.py --T1 $T1  --T2 $T2 --task2_start_from $task2_start_from --out_dir $out_dir \
                                        --type BR --from_task1 $from_task1
        echo "Submitted $i-th BR simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 3 ]; then
    echo "Simulation 3: Transfer Bayesian Optimization on Needle function"

    shift_task2="0.05"
    T1=40
    T2=20

    num_rep=20

    for shift in $shift_task2; do
        echo "Task 1: needle function; Task 2: needle function after shifting $shift"
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            mkdir -p $path_data/Needle_shift_${shift}/$i 
            out_dir=$path_data/Needle_shift_${shift}/$i

            job_name=Needle_shift_${shift}_${task2_start_from}_$i
            sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                            --type NEEDLE  --needle_shift ${shift}  --from_task1 $from_task1
            echo "Submitted $i-th Needle simulation by Slurm"
        done
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 4 ]; then
    echo "Simulation 4: Transfer Bayesian Optimization from Mono to Needle function"

    shift_task2="0.1"
    T1=20
    T2=20

    num_rep=20

    for shift in $shift_task2; do
        echo "Task 1: mono function; Task 2: needle function after shifting $shift"
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            mkdir -p $path_data/Mono2Needle_shift_${shift}/$i 
            out_dir=$path_data/Mono2Needle_shift_${shift}/$i

            job_name=Mono2Needle_shift_${shift}_${task2_start_from}_$i
            sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                            --type MONO2NEEDLE  --needle_shift ${shift}  --from_task1 $from_task1
            echo "Submitted $i-th Mono2Needle simulation by Slurm"
        done
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 5 ]; then
    echo "Simulation 5: Transfer Bayesian Optimization from Mono to Double exponential function"

    T1=20
    T2=20

    num_rep=20

    mu2="9"
    theta1="0.5" 
    theta2="2"

    echo "Task 1: mono exp function; Task 2: double exp function"
    for i in $(seq 1 $num_rep); do
        echo "Running $i-th simulation"
        mkdir -p $path_data/Mono2Double_mu2_${mu2}_theta2_$theta2/$i 
        out_dir=$path_data/Mono2Double_mu2_${mu2}_theta2_$theta2/$i
        job_name=Mono2Double_mu2_${mu2}_${task2_start_from}_$i
        sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                        --type MONO2DOUBLE  --from_task1 $from_task1
        echo "Submitted $i-th Mono2Double exponential simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 6 ]; then
    echo "Simulation 6: Transfer Bayesian Optimization from Double to Double exponential function"

    T1=20
    T2=20

    num_rep=20

    echo "Task 1: double exp function; Task 2: double exp function"
    for i in $(seq 1 $num_rep); do
        echo "Running $i-th simulation"
        mkdir -p $path_data/Double2Double_10sample_1bad_prior/$i
        out_dir=$path_data/Double2Double_10sample_1bad_prior/$i
        job_name=Double2Double_bad_prior_$i
        sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                        --type DOUBLE2DOUBLE  --from_task1 $from_task1
        echo "Submitted $i-th Double2Double exponential simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 7 ]; then
    echo "Simulation 7: Transfer Bayesian Optimization from Triple to Double exponential function"

    T1=20
    T2=20

    num_rep=20

    echo "Task 1: triple exp function; Task 2: double exp function"
    for i in $(seq 1 $num_rep); do
        echo "Running $i-th simulation"
        mkdir -p $path_data/Triple2Double_sample_far2_prior/$i 
        out_dir=$path_data/Triple2Double_sample_far2_prior/$i
        job_name=Triple2Double_far2_${task2_start_from}_$i
        sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                        --type TRIPLE2DOUBLE  --from_task1 $from_task1
        echo "Submitted $i-th Triple2Double exponential simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 8 ]; then
    echo "Simulation 8: Transfer Bayesian Optimization from Double to Triple exponential function"

    T1=20
    T2=20

    num_rep=20

    echo "Task 1: double exp function; Task 2: triple exp function"
    for i in $(seq 1 $num_rep); do
        echo "Running $i-th simulation"
        mkdir -p $path_data/Double2Triple/$i 
        out_dir=$path_data/Double2Triple/$i
        job_name=Double2Triple_${task2_start_from}_$i
        sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                        --type DOUBLE2TRIPLE  --from_task1 $from_task1
        echo "Submitted $i-th Double2Triple exponential simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 9 ]; then
    echo "Simulation 9: Transfer Bayesian Optimization from 2D Triple to 2D Triple exponential function"

    T1=20
    T2=20

    num_rep=20

    echo "Task 1: 2D Triple exp function; Task 2: 2D Triple exp function"
    for i in $(seq 1 $num_rep); do
        echo "Running $i-th simulation"
        mkdir -p $path_data/2D_Triple2Triple_20sample_2bad_prior_scaleTheta/$i
        out_dir=$path_data/2D_Triple2Triple_20sample_2bad_prior_scaleTheta/$i
        job_name=Triple2Triple_2D_${task2_start_from}_$i
        sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                        --type TRIPLE2TRIPLE_2D  --from_task1 $from_task1
        echo "Submitted $i-th 2D Triple2Triple exponential simulation by slurm"
    done
fi

