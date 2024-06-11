# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List, Tuple, Callable, Dict, Hashable
from functools import partial
import pandas as pd
import numpy as np
import os, copy

from emukit.core import ParameterSpace
from GPy.kern import RBF

from transfergpbo.models import (
    TaskData,
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
    STBO,
)
from transfergpbo.bo.run_bo import run_bo
from transfergpbo import models, benchmarks

## normal 
#from transfergpbo.parameters import parameters as params

## Exp1: double2double 1D 
from transfergpbo.parameters_d2d_1d import parameters as params

## Exp2: double2triple 1D
#from transfergpbo.parameters_d2t_1d import parameters as params

## Exp3: triple2triple 2D
#from transfergpbo.parameters_t2t_2d import parameters as params

## EXP4：exponential 2D
#from transfergpbo.parameters_exp_2d import parameters as params

## EXP5: XGB from Boston to California
#from transfergpbo.parameters_xgb_5d import parameters as params

def generate_functions(
    function_name: str,
    num_source_functions: int = 1,
    params_source: List[Dict[str, float]] = None,
    params_target: Dict[str, float] = None,
) -> Tuple[Callable, List[Callable], ParameterSpace]:
    """Generate the source and target functions from the respective family."""
    function = getattr(benchmarks, function_name)
    fun_target, space = (
        function() if params_target is None else function(**params_target)
    )
    funs_source = []
    for i in range(num_source_functions):
        fun, _ = function() if params_source is None else function(**params_source[i])
        funs_source.append(fun)
        
    return fun_target, funs_source, space


def get_benchmark(
    benchmark_name: str,
    num_source_points: List[int],
    output_noise: float = 0.0,
    params_source: List[Dict[str, float]] = None,
    params_target: Dict[str, float] = None,
    dir_source: str = None,
    file_source: str = None,
    num_rep: int = None,
) -> Tuple[Callable, Dict[Hashable, TaskData], ParameterSpace]:
    """Create the benchmark object."""
    num_source_functions = len(num_source_points)

    f_target, f_source, space = generate_functions(
        benchmark_name, num_source_functions, params_source, params_target
    )
    print("source func: ", f_source)
    print("target func: ", f_target)

    source_data = {}
    if dir_source == None:  # generate if no existed source data
        for i, (n_source, f) in enumerate(zip(num_source_points, f_source)):
            rand_points = space.sample_uniform(point_count=n_source)
            source_data[i] = TaskData(
                X=rand_points, Y=f(rand_points, output_noise=output_noise)
            )
    else:
        file_source_num_rep = os.path.join(dir_source, str(num_rep), file_source)

        df = pd.read_csv(file_source_num_rep, sep='\t', comment='#', header=None, skiprows=1)
        Y_tmp = -1*df[0].values
        Y = Y_tmp.reshape(-1, 1)
        X = df.iloc[:, 1:].values

        # only support 1 source now
        source_data[0] = TaskData(
            X=X, Y=Y
        )

    return f_target, source_data, space


def get_model(
    model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData]
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP or model_class == STBO:
        model = model_class(space.dimensionality)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)

    return model

def best_source_point(source_data: Dict[Hashable, TaskData]):
    """return best point from source task"""
    data = copy.deepcopy(source_data)
    best_X_source = {}

    for i, (source_id, source_d) in enumerate(data.items()):
        X_i = source_d.X
        Y_i = source_d.Y
        idx_min = np.argmin(Y_i)
        best_X_i = X_i[idx_min]
        best_X_source[i] = best_X_i
    
    return best_X_source

def run_experiment(parameters: dict) -> List[float]:
    """The actual experiment code."""
    num_source_points = parameters["benchmark"]["num_source_points"]
    technique = parameters["technique"]
    benchmark_name = parameters["benchmark"]["name"]
    num_steps = parameters["benchmark"]["num_steps"]

    output_noise = parameters["output_noise"]
    start_bo = parameters["start_bo"]    
    params_source = parameters["benchmark"].get("parameters_source", None)
    params_target = parameters["benchmark"].get("parameters_target", None)

    # start from existed f1 task file
    num_repetitions = parameters["benchmark"]["num_repetitions"]
    dir_source_points = parameters["benchmark"]["dir_source_points"]
    file_source_points = parameters["benchmark"]["file_source_points"]

    # output f2 results
    dir_target_points = parameters["benchmark"]["dir_target_points"]

    if "task1_gp" in file_source_points:
        file_target_points = parameters["benchmark"]["file_target_points"] + '_' + technique + "_from_gp.tsv"
    elif "task1_rand" in file_source_points:
        file_target_points = parameters["benchmark"]["file_target_points"] + '_' + technique + "_from_rand.tsv"        

    regret_total = []
    for num_rep in range(1, num_repetitions+1):
        # Initialize the benchmark and model
        f_target, source_data, space = get_benchmark(
            benchmark_name, num_source_points, output_noise, params_source, params_target,
            dir_source_points, file_source_points, num_rep
        )
        model = get_model(technique, space, source_data)

        # Run BO and write parameters for real experiments
        best_X_source = best_source_point(source_data)

        # Run BO and return the regret
        regret_num_rep = run_bo(
            experiment_fun=partial(f_target, output_noise=output_noise),
            model=model,
            space=space,
            best_X_source=best_X_source,
            start_bo=start_bo,
            num_iter=num_steps,
            noiseless_fun=partial(f_target, output_noise=0.0),
            dir_target_points=dir_target_points,
            file_target_points=file_target_points, 
            num_rep=num_rep
        )

        regret_total.append(regret_num_rep)
        print("The " + str(num_rep) + "-th repetition is finished !")
    
    return regret_total


if __name__ == "__main__":
    run_experiment(params)

