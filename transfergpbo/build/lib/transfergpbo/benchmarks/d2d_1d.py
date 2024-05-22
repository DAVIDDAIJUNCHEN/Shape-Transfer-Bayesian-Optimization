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


def d2d_1d_function(x, alpha1: float = 1.0, alpha2: float = 1.5, 
                    output_noise: float = 0.0):
    x = np.asarray(x)
    y = alpha1 * np.exp(- 0.5* np.square(x)) + alpha2 * np.exp(-0.5 * np.square(x - 5))
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    y = -1* y
    return y


def d2d_1d(alpha1: float = None, alpha2: float = None) -> Tuple[Callable, ParameterSpace]:
    if alpha1 is None:
        alpha1 = np.random.uniform(low=0.5, high=2.0)
    if alpha2 is None:
        alpha2 = np.random.uniform(low=0.5, high=2.0)
    
    return partial(
        d2d_1d_function, alpha1=alpha1, alpha2=alpha2
    ), ParameterSpace(
        [
            ContinuousParameter("x", -5.0, 10.0),
        ]
    )
