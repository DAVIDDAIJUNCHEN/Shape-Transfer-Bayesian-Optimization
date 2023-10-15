 #!/usr/bin/env bash

stage=$1       # 0: run both EXP and BR; 1: run EXP only; 2: run BR only
from_task1=$2  # 1: start from task1, 0: start from task2
task2_start_from=$3

path_data=data/


# check arguments
if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then 
    echo "Usage: bash run_simulation.sh <stage-number> <start_from_task1>" 
    echo "<stage-number>: 0, run all simulation;"
    echo "<stage-number>: 1, run EXP only;"
    echo "<stage-number>: 2, run Branin only;"
    echo "<stage-number>: 3, run Needle only;"
    echo "<start_from_task1>: 1, run from task1;"
    echo "<start_from_task1>: 0, skip task1 and run task 2;" 
    echo "<task2_start_from>: gp or rand ." && exit 0
fi


if [ $stage -eq 0 ] || [ $stage -eq 1 ]; then 
    echo "Simulation 1: Transfer Bayesian Optimization on Exponential target function"

    Thetas="1.414"
    mu_1="0_0"
    mu_2="0.707_0.707"

    T1=20
    T2=20

    num_rep=20

    for theta in $Thetas; do
        echo "Task 1: mean=$mu_1, theta=$theta; Taks 2: mean=$mu_2, theta=$theta, starts from best point in $task2_start_from" 
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            mkdir -p $path_data/EXP_mu2_${mu_2}_theta_$theta/$i
            out_dir=$path_data/EXP_mu2_${mu_2}_theta_$theta/$i

            job_name=EXP_mu2_${mu_2}_theta_$theta_$i
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

        job_name=Branin_$i
        sbatch --job-name=$job_name ./main_simulation.py --T1 $T1  --T2 $T2 --task2_start_from $task2_start_from --out_dir $out_dir --type BR --from_task1 $from_task1
        echo "Submitted $i-th BR simulation by Slurm"
    done
fi


if [ $stage -eq 0 ] || [ $stage -eq 3 ]; then
    echo "Simulation 3:  Transfer Bayesian Optimization on Needle function"

    shift_task2="0.01 0.05 0.3 1"
    T1=20
    T2=20

    num_rep=20

    for shift in $shift_task2; do
        echo "Task 1: needle function; Task 2: needle function after $shift shiftting"
        for i in $(seq 1 $num_rep); do
            echo "Running $i-th simulation"
            mkdir -p $path_data/Needle_shift_${shift}/$i 
            out_dir=$path_data/Needle_shift_${shift}/$i

            job_name=Needle_shift_${shift}_$i
            sbatch --job-name=$job_name ./main_simulation.py  --T1 $T1  --T2 $T2  --task2_start_from $task2_start_from  --out_dir $out_dir \
                                            --type NEEDLE  --needle_shift ${shift}  --from_task1 $from_task1
            echo "Submitted $i-th Needle simulation by Slurm"
        done
    done
fi

