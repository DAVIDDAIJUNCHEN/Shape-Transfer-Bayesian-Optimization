# Shape-Transfer-Bayesian-Optimization

## Simulation Instruction

We have two types of simulation experiments. The first type takes exponential functions as target functions in task 1 and 2. The second type takes Branin function as target function of task 1, and modified Branin function as target function of task 2.

### type 1:

$$
f_1(\bold{x}) = \exp\{-\frac{1}{2\theta^2}\|\bold{x}-\bold{\mu}_1\|^2\}
$$

$$
f_2(\bold{x}) = \exp\{-\frac{1}{2\theta^2}\|\bold{x}-\mu_2\|^2\}
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
