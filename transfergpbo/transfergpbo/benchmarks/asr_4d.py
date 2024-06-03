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
import pandas as pd


def asr_4d(language: str = None) -> Tuple[ParameterSpace]:
    if language is None:
        language = "yue-CHN"

    return ParameterSpace(
        [
            ContinuousParameter("x1", 0.0, 5000),
            ContinuousParameter("x2", 0.0, 5000),
            ContinuousParameter("x3", 0.0, 5000),
            ContinuousParameter("x4", 0.0, 5000),
        ]
    )

