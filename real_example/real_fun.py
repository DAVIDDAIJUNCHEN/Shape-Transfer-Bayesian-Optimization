"""
Authors: Daijun Chen

Implementation of all the testing functions
"""

import numpy as np
from numpy import *
import math
from numpy.matlib import *
from scipy.stats import multivariate_normal
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class XGB_Calif():
    """
    XGBoost trained/tested on California housing data
    """
    def __init__(self, noisy=False):
        self.noisy = noisy
        self.dim=5
        self.max=10

        self.bounds=np.array([
                            [2, 15],        # max_depth
                            [0.01, 0.3],    # learning_rate
                            [0, 10],        # max_delta_step
                            [0, 1],         # colsample_bytree
                            [0, 1],         # subsample
                            # [1, 20],        # min_child_weight
                            # [0, 10],        # gamma
                            # [0, 10],        # reg_alpha
                                ])

        housing = fetch_california_housing()
    
        X = housing.data
        Y = housing.target
    
        self.data_dmatrix = xgb.DMatrix(data=X, label=Y)

    def __call__(self, x):
        max_depth, lr, max_delta_step, colsample_bytree, subsample = x

        params =  {
                    'objective': 'reg:squarederror',
                    'max_depth': int(max_depth),
                    'learning_rate': lr,
                    'max_delta_step': int(max_delta_step),
                    'colsample_bytree': colsample_bytree,
                    'subsample' : subsample
                    }

        cv_results = xgb.cv(params=params, 
                            dtrain=self.data_dmatrix, 
                            nfold=3, 
                            seed=3,
                            num_boost_round=50000,
                            early_stopping_rounds=50,
                            metrics='rmse')

        return 10 - cv_results['test-rmse-mean'].min()


class XGB_Boston():
    """
    XGBoost trained/tested on Boston
    """
    def __init__(self, noisy=False):
        self.noisy = noisy
        self.dim=5
        self.max=10

        self.bounds=np.array([
                            [2, 15],        # max_depth
                            [0.01, 0.3],    # learning_rate
                            [0, 10],        # max_delta_step
                            [0, 1],         # colsample_bytree
                            [0, 1],         # subsample
                            # [1, 20],        # min_child_weight
                            # [0, 10],        # gamma
                            # [0, 10],        # reg_alpha
                                ])

        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        Y = raw_df.values[1::2, 2]

        self.data_dmatrix = xgb.DMatrix(data=X, label=Y)

    def __call__(self, x):
        max_depth, lr, max_delta_step, colsample_bytree, subsample = x

        params =  {
                    'objective': 'reg:squarederror',
                    'max_depth': int(max_depth),
                    'learning_rate': lr,
                    'max_delta_step': int(max_delta_step),
                    'colsample_bytree': colsample_bytree,
                    'subsample' : subsample
                    }

        cv_results = xgb.cv(params=params, 
                            dtrain=self.data_dmatrix, 
                            nfold=3, 
                            seed=3,
                            num_boost_round=50000,
                            early_stopping_rounds=50,
                            metrics='rmse')

        return 10 - cv_results['test-rmse-mean'].min()


class XGB_Tornodo():
    """
    XGBoost trained/tested on Tornodo
    """
    def __init__(self, noisy=False):
        self.noisy = noisy
        self.dim=5
        self.max=10

        self.bounds=np.array([
                            [2, 15],        # max_depth
                            [0.01, 0.3],    # learning_rate
                            [0, 10],        # max_delta_step
                            [0, 1],         # colsample_bytree
                            [0, 1],         # subsample
                            # [1, 20],        # min_child_weight
                            # [0, 10],        # gamma
                            # [0, 10],        # reg_alpha
                                ])

        scaler = StandardScaler()
        X = pd.read_csv("Toronto_Housing_Market/Tornodo_X.csv")
        Y = pd.read_csv("Toronto_Housing_Market/Tornodo_Y.csv")
        
        X = X.values
        Y = Y.values

        Y = Y[:,3]
        Y = Y.reshape(-1, 1)
        Y = scaler.fit_transform(Y)

        self.data_dmatrix = xgb.DMatrix(data=X, label=Y)

    def __call__(self, x):
        max_depth, lr, max_delta_step, colsample_bytree, subsample = x

        params =  {
                    'objective': 'reg:squarederror',
                    'max_depth': int(max_depth),
                    'learning_rate': lr,
                    'max_delta_step': int(max_delta_step),
                    'colsample_bytree': colsample_bytree,
                    'subsample' : subsample
                    }

        cv_results = xgb.cv(params=params, 
                            dtrain=self.data_dmatrix, 
                            nfold=3, 
                            seed=3,
                            num_boost_round=50000,
                            early_stopping_rounds=50,
                            metrics='rmse')

        return 10 - cv_results['test-rmse-mean'].min()



if __name__ == "__main__":
    """ Traditional Method: Run one by one"""
    x = (3, 0.05, 2, 0.5, 0.3)
    max_depth, lr, max_delta_step, colsample_bytree, subsample = x

    # California House Price
    response_cali = XGB_Calif(noisy=False)
    print(response_cali(x))

    # Boston House Price
    response_boston = XGB_Boston(noisy=False)
    print(response_boston(x))

    # Tornodo House Price
    response_tornodo = XGB_Tornodo(noisy=False)
    print(response_tornodo(x))

