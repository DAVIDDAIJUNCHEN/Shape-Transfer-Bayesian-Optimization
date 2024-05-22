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

from typing import Tuple, Callable
from functools import partial

import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd


def xgb_5d_function(
    x: np.ndarray,
    city: str = "Boston",
    output_noise: float = 0.0):

    # hyper parameter
    max_depth, lr, max_delta_step, colsample_bytree, subsample = x[0]

    params =  {
                'objective': 'reg:squarederror',
                'max_depth': int(max_depth),
                'learning_rate': lr,
                'max_delta_step': int(max_delta_step),
                'colsample_bytree': colsample_bytree,
                'subsample' : subsample
                }

    # training data
    if city == "Boston" or city == "boston":
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        Y = raw_df.values[1::2, 2]
        data_dmatrix = xgb.DMatrix(data=X, label=Y)        
    elif city == "California" or city == "california":
        housing = fetch_california_housing()
        X = housing.data
        Y = housing.target
        Y = np.array([y*10 for y in Y])
        data_dmatrix = xgb.DMatrix(data=X, label=Y)

    # compute cross validation error
    cv_results = xgb.cv(params=params, 
                        dtrain=data_dmatrix, 
                        nfold=3, 
                        seed=3,
                        num_boost_round=50000,
                        early_stopping_rounds=50,
                        metrics='rmse')    

    y = 10 - cv_results['test-rmse-mean'].median()
    y = -1*y
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    
    return np.array([[y]])


def xgb_5d(city: str = None) -> Tuple[Callable, ParameterSpace]:
    if city is None:
        city = "Boston"
    
    return partial(xgb_5d_function, 
                   city=city
    ), ParameterSpace(
        [
            ContinuousParameter("x1", 2.0, 15.0),
            ContinuousParameter("x2", 0.01, 0.3),
            ContinuousParameter("x3", 0.01, 10.0),
            ContinuousParameter("x4", 0.01, 1.0),
            ContinuousParameter("x5", 0.01, 1.0),
        ]
    )


if __name__ == "__main__":
    x = [[8.24127596005908,        0.1820006829315179  ,    4.945703487432738   ,    1.0   ,  0.0]]
    
    print(xgb_5d_function(x, city="California"))
