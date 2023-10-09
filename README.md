# Shape-Transfer-Bayesian-Optimization

## Simulation Instruction

We have two types of simulation experiments. The first type takes exponential functions as target functions in task 1 and 2. The second type takes Branin function as target function of task 1, and modified Branin function as target function of task 2.

### type 1:

$$
f_1(x) = \exp\{-\frac{1}{2\theta^2}\|x-\mu_1\|^2\}
$$

$$
f_2(x) = \exp\{-\frac{1}{2\theta^2}\|x-\mu_2\|^2\}
$$

To run the simulation, you can change the configuration given in the ``run_simulation.sh`` script.

```
Thetas="1"  
mu_1="0_0"
mu_2="2_2"

T1=20          # number of experiment points in task1 
T2=20          # number of experiment points in task2

num_rep=20     # repetition times 
```

Where Thetas is the list of $\theta$ values seprated by space, `mu_1` and `mu_2` are the $\mu_1$ and $\mu_2$ vectors whose components are separated by underline _ .

Then, run script `run_simulation.sh` with 3 parameters (stage, from_task1, task2_start_from) as follows,

```
./run_simulation.sh  <stage-number>   <start_from_task1>   <task2_start_from>
```

where both exponential and Branin types simulations will be run if `stage-number` is 0, only exponential simulation will be executed if `stage-number` is 1, and only Branin simulation will be executed if `stage-number` is 2.

When the task1 experiments have been done,  task1 result files can be found in ./data dir. If you want to skip task1 experiments, please set `start_from_task1` 0.

In task1, normal Gaussian process model and random search have been applied. Therefore, you can start task2 experiments based on Gaussian process or random search results in task1. If `task2_start_from` is `gp`, then task2 starts from Gaussian process results in task1; if `task2_start_from` is `rand`, then task2 starts from random search results in task1.
