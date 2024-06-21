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


def exp_2d_function(
    x: np.ndarray, 
    theta: float = 0.87, mu: np.ndarray = np.array([[0.435, 0.435]]), 
    output_noise: float = 0.0):

    mu = np.asarray(mu)
    x = np.asarray(x)
    
    diff_norm2 = np.sum(np.square(x - mu))

    y = np.exp(-0.5*diff_norm2/np.square(theta))
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    y = -1*y
    return np.array([[y]])


def exp_2d(theta: float = None, mu: np.ndarray = None) -> Tuple[Callable, ParameterSpace]:
    if theta is None:
        theta = np.random.uniform(low=0.5, high=2.0)
    
    if mu is None:
        mu = np.array([np.random.uniform(low=0, high=3.0, size=2)])

    return partial(exp_2d_function,
                   theta=theta, mu=mu 
    ), ParameterSpace(
        [
            ContinuousParameter("x1", -1, 3),
            ContinuousParameter("x2", -1, 3),
        ]
    )
