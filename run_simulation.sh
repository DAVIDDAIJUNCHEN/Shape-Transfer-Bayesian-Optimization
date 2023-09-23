#!/bin/env bash

# Part 1: Exponential target function
Thetas="0.25 0.5 1.0 2.0 3.0"
mu_1="0.0"
mu_2="1.0"

T1=10
T2=4

num_rep=1

for i in seq 1 $num_rep; do
    for theta in $Thetas; do
        python ./main_simulation.py  --T1 $T1  --T2 $T2  --type EXP  --mu1 $mu_1  --mu2 $mu_2  --theta $theta 
    done
done 



# Part 2: Branni & Modified Branni function 
T1=10
T2=4

num_rep=1

for i in seq 1 $num_rep; do 
    python ./main_simulation.py --T1 $T1  --T2 $T2 --type BR 

done

