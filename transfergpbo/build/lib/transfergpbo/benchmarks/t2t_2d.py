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


def t2t_2d_function(
    x: np.ndarray, 
    alpha1: float = 1.0, beta1: np.ndarray = np.array([[0.0, 0.0]]),
    alpha2: float = 1.4, beta2: np.ndarray = np.array([[5.0, 5.0]]),
    alpha3: float = 1.9, beta3: np.ndarray = np.array([[10.0, 10.0]]), 
    output_noise: float = 0.0):

    beta1 = np.asarray(beta1)
    beta2 = np.asarray(beta2)
    beta3 = np.asarray(beta3)

    x = np.asarray(x)
    
    diff1_norm2 = np.sum(np.square(x - beta1))
    diff2_norm2 = np.sum(np.square(x - beta2))
    diff3_norm3 = np.sum(np.square(x - beta3))

    y = alpha1*np.exp(-0.5*diff1_norm2) + alpha2*np.exp(-0.5*diff2_norm2) + alpha3*np.exp(-0.5*diff3_norm3)
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    y = -1*y
    
    return np.array([[y]])


def t2t_2d(alpha1: float = None, beta1: np.ndarray = None, 
           alpha2: float = None, beta2: np.ndarray = None,
           alpha3: float = None, beta3: np.ndarray = None) -> Tuple[Callable, ParameterSpace]:
    if alpha1 is None:
        alpha1 = np.random.uniform(low=0.5, high=2.0)
    if alpha2 is None:
        alpha2 = np.random.uniform(low=1.0, high=2.0)
    if alpha3 is None:
        alpha3 = np.random.uniform(low=0.5, high=2.0)
    
    if beta1 is None:
        beta1 = np.array([np.random.uniform(low=-1.0, high=1.0, size=2)])
    if beta2 is None:
        beta2 = np.array([np.random.uniform(low= 4.0, high=6.0, size=2)])
    if beta3 is None:
        beta3 = np.array([np.random.uniform(low=9.0, high=11.0, size=2)])

    return partial(t2t_2d_function, 
                   alpha1=alpha1, beta1=beta1,
                   alpha2=alpha2, beta2=beta2,
                   alpha3=alpha3, beta3=beta3
    ), ParameterSpace(
        [
            ContinuousParameter("x1", -5.0, 15.0),
            ContinuousParameter("x2", -5.0, 15.0),
        ]
    )
