# Copyright (c) 2024 Daijun Chen
# 
# This code implements the Shape Transfer Bayesian Optimization (STBO) 

import copy
import numpy as np
from typing import Dict, Hashable, Union, Sequence, Tuple

from GPy.kern import RBF

from transfergpbo.models import InputData, TaskData, Model, GPBO



class STBO(Model):
    """Shape Transfer Bayesian Optimization (STBO)
    
    Transfer Learning model based on [Li et al: Shape Transfer Bayesian Optimization]().
    Given a source data set, the transfer to the target data set is done by training a 
    source GP on source data set, and training a diff-GP on difference data set (target value
    - source GP value on target data set). Note that MHGP also utlizes the same idea in 
    target mean function, but we differe much on the target variance.
    """

    def __init__(self, n_features: int, within_model_normalie: bool = False):
        """Initialize the method
        
        Args:
            n_features: Number of input parameters of the data
            within_model_normalize: Normalize each GP internally to imporve
            numerical stability
        """
        super().__init__()
        
        self.n_samples = 0
        self.n_features = n_features

        self._within_model_normalize = False

        self.source_gps = []

        # GP on difference between target data and source data
        self.target_gp = GPBO(
            RBF(self.n_features, ARD=True),
            noise_variance=0.0, # origin: 0.1
            normalize=False, #self._within_model_normalize,
        )

    def _compute_residuals(self, data: TaskData) -> np.ndarray :
        """Determine the difference between given y-values and the sum of predicted
        values from the models in 'source_gps'.

        Args:
            data: Observation (input and target) data.
                Input data: ndarray, `shape = (n_points, n_features)`
                Target data: ndarray, `shape = (n_points, 1)`

        Returns:
            Difference between observed values and sum of predicted values
            from `source_gps`. `shape = (n_points, 1)`
        """    
        if self.n_features != data.X.shape[1]:
            raise ValueError("Number of features in model and input data mismatch.")
        
        if not self.source_gps:
            return data.Y

        # the sum of all source GPs
        predicted_y = self.predict_posterior_mean(
            InputData(data.X), idx=(len(self.source_gps)-1)
        )

        residuals = data.Y - predicted_y

        return residuals 
    
    def meta_fit(
        self,
        source_datasets: Dict[Hashable, TaskData],
        optimize: Union[bool, Sequence[bool]] = False,
    ):
        """Train the source GPs on the given source data sets.

        Args:
            source_datasets: Dictionary containing the source datasets. The stack of GPs
                are trained on the residuals between two consecutive data sets in this
                list.
            optimize: Switch to run hyperparameter optimization.
        """
        # get optimize_flat: [optimize_1, ..., optimize_N]
        optimize = False
        assert isinstance(optimize, bool) or isinstance(optimize, list)

        if isinstance(optimize, list):
            assert len(source_datasets) == len(optimize)
        
        optimize_flag = copy.copy(optimize)

        if isinstance(optimize_flag, bool):
            optimize_flag = [optimize_flag] * len(source_datasets)

        for i, (source_id, source_data) in enumerate(source_datasets.items()):
            new_gp = self._meta_fit_single_gp(
                source_data,
                optimize=optimize_flag[i],
            )
            self._update_meta_data(new_gp)

    def _update_meta_data(self, *gps: GPBO):
        """Cache the meta data after meta training."""
        for gp in gps:
            self.source_gps.append(gp)

    def _meta_fit_single_gp(
        self,
        data: TaskData,
        optimize: bool,
    ) -> GPBO:
        """Train a new source GP on `data`.

        Args:
            data: The source dataset.
            optimize: Switch to run hyperparameter optimization.

        Returns:
            The newly trained GP.
        """
        residuals = self._compute_residuals(data)
        kernel = RBF(self.n_features, ARD=True)
        new_gp = GPBO(
            kernel, noise_variance=0.0, normalize=False
        ) # origin: noise_variance=0.1
        new_gp.fit(
            TaskData(X=data.X, Y=residuals),
            optimize
        )

        return new_gp  

    def fit(self, data: TaskData, optimize: bool=False):
        """Build target gp based on residual data"""
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )
        optimize = False
        self._X = copy.deepcopy(data.X)
        self._y = copy.deepcopy(data.Y)

        self.n_samples, n_features = self._X.shape
        if self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")
        
        residuals = self._compute_residuals(data)
            
        self.target_gp.fit(TaskData(data.X, residuals), optimize)        

    def predict(
        self, data: InputData, return_full: bool = False, with_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance based on STBO method"""
        if not self.source_gps:
            raise ValueError(
                "Error: source gps are not trained. Forgot to call `meta_fit`."
            )

        # returned mean: sum of means of the predictions of all source and target GPs
        mu = self.predict_posterior_mean(data)        

        # returned variance is the variance of target GP
        _, var_res_gp = self.target_gp.predict(
            data, return_full=return_full, with_noise=with_noise
        )        

        return mu, var_res_gp

    def predict_posterior_mean(self, data: InputData, idx: int = None) -> np.ndarray:
        """Predict the mean function for given test point(s).

        For `idx=None` returns the same as `self.predict(data)[0]` but avoids the
        overhead coming from predicting the variance. If `idx` is specified, returns
        the sum of all the means up to the `idx`-th GP. 

        Args:
            data: Input data to predict on.
                Data is provided as ndarray with shape = (n_points, n_features).
            idx: Integer of the GP in the stack. Counting starts from the bottom at
                zero. If `None`, the mean prediction of source and target GPs is returned.

        Returns:
            Predicted mean for every input. `shape = (n_points, 1)`        

        """
        all_gps = self.source_gps + [self.target_gp]

        # idx==None ==> return sum of all source and target GPs 
        if idx == None:
            idx = len(all_gps) - 1

        mu = np.zeros((data.X.shape[0], 1))

        for model in all_gps[: (idx + 1)]:
            mu += model.predict_posterior_mean(data)

        return mu

    def predict_posterior_covariance(self, x1: InputData, x2: InputData) -> np.ndarray:
        """Posterior covariance between two inputs.

        Args:
            x1: First input to be queried. `shape = (n_points_1, n_features)`
            x2: Second input to be queried. `shape = (n_points_2, n_features)`

        Returns:
            Posterior covariance at `(x1, x2)`. `shape = (n_points_1, n_points_2)`
        """
        return self.target_gp.predict_posterior_covariance(x1, x2)


