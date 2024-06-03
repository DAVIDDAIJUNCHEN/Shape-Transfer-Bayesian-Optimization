from typing import List, Tuple, Callable, Dict, Hashable
from functools import partial
import pandas as pd
import numpy as np
import os

from emukit.core import ParameterSpace
from GPy.kern import RBF

from transfergpbo.models import (
    TaskData,
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)

from transfergpbo.bo.run_bo import run_bo_interactive
from transfergpbo import models, benchmarks

# Real Example: ASR MNC to CAC
from transfergpbo.parameters_asr_4d import parameters as params


def generate_space_target(
    function_name: str,
    params_target: Dict[str, float] = None,
) -> Tuple[ParameterSpace]:
    """Generate space of target function (function_name)"""
    function = getattr(benchmarks, function_name)
    space = (
        function() if params_target is None else function(**params_target)
    )

    return space

def get_benchmark(
    benchmark_name: str,
    output_noise: float = 0.0,
    params_target: Dict[str, float] = None,
    dir_source: str = None,
    file_source: str = None,
    num_rep: int = None,
) -> Tuple[Dict[Hashable, TaskData], ParameterSpace]:
    """Create the benchmark object."""
    space = generate_space_target(benchmark_name, params_target)

    source_data = {}

    file_source_num_rep = os.path.join(dir_source, str(num_rep), file_source)

    df = pd.read_csv(file_source_num_rep, sep='\t', comment='#', header=None, skiprows=1)
    Y_tmp = -1*df[0].values
    Y = Y_tmp.reshape(-1, 1)
    X = df.iloc[:, 1:].values

    # only support 1 source now
    source_data[0] = TaskData(
        X=X, Y=Y
    )

    return source_data, space

def get_model(
    model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData]
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)

    return model

def run_experiment_interactive(parameters: dict) -> List[float]:
    "The real experiment code with interaction"

    # Step 1: Read parameters from config 
    # 1.1 general parameters
    num_source_points = parameters["benchmark"]["num_source_points"]    
    technique = parameters["technique"]
    benchmark_name = parameters["benchmark"]["name"]    

    output_noise = parameters["output_noise"]
    params_source = parameters["benchmark"].get("parameters_source", None)    
    params_target = parameters["benchmark"].get("parameters_target", None)    

    # 1.2 input parameters
    num_repetitions = parameters["benchmark"]["num_repetitions"]
    dir_source_points = parameters["benchmark"]["dir_source_points"]
    file_source_points = parameters["benchmark"]["file_source_points"]

    # 1.3 output parameters
    dir_target_points = parameters["benchmark"]["dir_target_points"]

    if "task1_gp" in file_source_points:
        file_target_points = parameters["benchmark"]["file_target_points"] + '_' + technique + "_from_gp.tsv"
    elif "task1_rand" in file_source_points:
        file_target_points = parameters["benchmark"]["file_target_points"] + '_' + technique + "_from_rand.tsv"           

    # Step 2: Get source data & build f1 model
    for num_rep in range(1, num_repetitions+1):
        source_data, space = get_benchmark(
            benchmark_name, output_noise, params_target, dir_source_points, file_source_points, num_rep 
        )
        model = get_model(technique, space, source_data)
    
        # Run BO and write parameters for real experiments
        run_bo_interactive(
           model=model,
           space=space,
           dir_target_points=dir_target_points,
           file_target_points=file_target_points,
           num_rep=num_rep
        )

    return 0


if __name__ == "__main__":
    run_experiment_interactive(params)

