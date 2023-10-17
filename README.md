# Shape-Transfer-Bayesian-Optimization

## Usage

To apply STBO method in real tasks, you can easily run `./main.py` script to find next point for your tasks. Assume you have two tasks to optimize their hyperparameters, and these two tasks have the same hyperparameters.

### task 1:

In task 1, you can use EI (Expected Improvement) method based on GP (Gaussian Process) Model,

* step 1: add initial experiment points

  Before start the EI method, you need to prepare some initial experiment points and get the responses by running task 1 on them. after that, add these initialization points to a file by following the header below, e.g. , `./data/experiment_points_task1_gp.tsv`.

  ```
  response#dim1#dim2#dim3
  0.4#3#4#5
  0.5#4#5#6
  ```
* step 2: run `./main.py` with `task=1`

  Let variable `task = 1` in `./main.py` script, then you can get the next experiment point for your task 1 after run `./main.py` .
* step 3: run task 1

  Get the task 1 response of new experiment point from step 2, and add the new experiment point and its response into file created in step 1, e.g. , `./data/experiment_points_task1_gp.tsv`.

Iteratively execute step 2 and step 3, you can do as many task 1 experiments as you can afford. At the same time, the best point (with largest response value) can be selected from task 1 experiment file, e.g. , `./data/experiment_points_task1_gp.tsv`.

### task 2:

In task 2, three optimizing hyper parameter methods have been prepared, STBO (Shape Transfer Bayesian Optimization), BCBO (Biases  Corrected Bayesian Optimization) and EI from best point. Among these three methods, EI from best point is exactly the same as EI used in task 1, only the initial points differ (EI in task 2 starts from best point selected in task 1). In addition, not only STBO and BCBO start from best point in task 1, these two methods transfer surrogate model knowledge from task 1 to task 2.

* step 1: add initial experiment points

  Although any initial experiment points can be used in task 2, it is recommended that starts from best point in task 1. Add the intial experiment points and corresponding responses into task 2 files, e.g., `./data/experiment_points_task2_STBO.tsv`, `./data/experiment_points_task2_BCBO.tsv` and `./data/experiment_points_task2_gp.tsv` . The format should be exactly the same as in task 1 file.
* step 2: run `./main.py` with `task=2`

  Let variable `task = 2` in `./main.py` script, then you can get the next experiment point for your task 2 after run `./main.py`. Note that, different method is going to give different next point for your task 2 experiment.
* step 3: run task 2

  For each next experiment point selected by these 3 methods, get the task 2 response, add the new experiment points and their responses into corresponding files created in step 1, e.g. , `./data/experiment_points_task2_STBO.tsv`, `./data/experiment_points_task2_BCBO.tsv` and `./data/experiment_points_task2_gp.tsv` .

Iteratively execute step 2 and step 3, you can run as many task 2 experiments as you can afford. Finally, the best point (with largest response value) can be selected from task 2 experiment files, which should be your optimized hyperparameters in task 2.

### Less task 2 experiment

In the above task 2 section, three transfer Bayesian optimization methods have been utilized. If you only want to use some of the three methods, just run task 2 on the next point selected by this method and only update the corresponding task 2 file.

## Simulation Instruction

We have three types of simulation experiments. The first type takes exponential functions as target functions in task 1 and 2. The second type takes Branin function as target function of task 1, and modified Branin function as target function of task 2.

### type 1:

$$
f_1(x) = \exp\{-\frac{1}{2\theta^2}\|x-\mu_1\|^2\}
$$

$$
f_2(x) = \exp\{-\frac{1}{2\theta^2}\|x-\mu_2\|^2\}
$$

#### configuration

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

#### run simulation

Then, run script `run_simulation.sh` with 3 parameters (stage, from_task1, task2_start_from) as follows,

```
./run_simulation.sh  <stage-number>   <start_from_task1>   <task2_start_from>
```

where both exponential and Branin types simulations will be run if `stage-number` is 0, only exponential simulation will be executed if `stage-number` is 1, and only Branin simulation will be executed if `stage-number` is 2.

When the task1 experiments have been done,  task1 result files can be found in ./data dir. If you want to skip task1 experiments, please set `start_from_task1` 0.

In task1, normal Gaussian process model and random search have been applied. Therefore, you can start task2 experiments based on Gaussian process or random search results in task1. If `task2_start_from` is `gp`, then task2 starts from Gaussian process results in task1; if `task2_start_from` is `rand`, then task2 starts from random search results in task1. e.g.

```
./run_simulation.sh  1   1   rand
```

the above command execute to run exponential simulation only from task1, and run task2 based on  task 1 random search results.

#### analyze results

After simulation jobs are finished, both task1 and 2 results can be found in `EXP_mu2_x_x_theta_x` subdir dir under `./data` , e.g. `./data/EXP_mu2_1.0_1.0_theta_0.5` . In this dir, `num_rep` subdirs can be found and each subdir contains simulations results. To analyze these simulation results, `./analyze_results.py` tool generate some plots.

```
python3 ./analyze_results.py  <input_dir>  <out_dir>  <topic>
```

where `input_dir` is the output dir after run simulation, `out_dir` is the analyzing output dir, and topic decides what files to analyze, possible candidates (`task1`,  `from_gp`, `from_rand`). e.g. the following command is running to analyze task2 results starting from random search (task1).

```
python3 ./analyze_results.py  ./data/EXP_mu2_1.0_1.0_theta_0.5 ./simulation_results/EXP_theta_0.5 from_rand
```
