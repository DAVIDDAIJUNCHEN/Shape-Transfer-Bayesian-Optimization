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


def t2t_1d_function(x, 
                    alpha1: float = 1.0, beta1: float = 0.0, 
                    alpha2: float = 1.4, beta2: float = 5.0,
                    alpha3: float = 1.9, beta3: float = 10.0, 
                    output_noise: float = 0.0):
    x = np.asarray(x)
    y = alpha1*np.exp(-0.5*np.square(x - beta1)) + alpha2*np.exp(-0.5*np.square(x - beta2)) + alpha3*np.exp(-0.5*np.square(x - beta3))
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    y = -1*y

    return y


def t2t_1d(alpha1: float = None, beta1: float = None, 
           alpha2: float = None, beta2: float = None,
           alpha3: float = None, beta3: float = None) -> Tuple[Callable, ParameterSpace]:
    if alpha1 is None:
        alpha1 = np.random.uniform(low=0.5, high=2.0)
    if alpha2 is None:
        alpha2 = np.random.uniform(low=0.0, high=2.0)
    if alpha3 is None:
        alpha3 = np.random.uniform(low=0.5, high=2.0)
    
    if beta1 is None:
        beta1 = np.random.uniform(low=0.0, high=1.0)
    if beta2 is None:
        beta2 = np.random.uniform(low=4.0, high=6.0)
    if beta3 is None:
        beta3 = np.random.uniform(low=9.0, high=11.0)

    return partial(t2t_1d_function, 
                   alpha1=alpha1, beta1=beta1,
                   alpha2=alpha2, beta2=beta2,
                   alpha3=alpha3, beta3=beta3
    ), ParameterSpace(
        [
            ContinuousParameter("x", -5.0, 15.0),
        ]
    )
