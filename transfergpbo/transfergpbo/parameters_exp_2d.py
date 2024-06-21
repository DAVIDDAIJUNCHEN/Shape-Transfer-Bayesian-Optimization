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

parameters = {
    # Mandatory parameter. Choose from: GPBO, MHGP, SHGP, BHGP, HGP, WSGP, MTGP, RGPE
    "technique": "STBO",   # 1. MTGP 2. WSGP 3. HGP 4. BHGP 5. SHGP 6. STBO 7. DiffGP 8. MHGP
    "benchmark": {
        # Mandatory parameter. Choose from: forrester, alpine, branin, hartmann3d,
        # hartmann6d
        "name": "exp_2d",
        # Mandatory parameter of type List[int]. Has no effect for GPBO
        "num_source_points": [20],
        "num_repetitions": 20, # 20
        "dir_source_points": "/mnt/users/daijun_chen/gits/github/Shape-Transfer-Bayesian-Optimization/data/EXP_mu2_2_2_theta_1",
        "file_source_points": "simExp_points_task1_rand.tsv",
        # Mandatory parameter. Defines the number of BO iterations.
        "num_steps": 20,
        # Optional parameter of type List[Dict[str, float]] or None. Defines the
        # parameters of the source functions. Leave to None for random sampling
        # according to default probabilities for the respective function family.
        "parameters_source": [{"theta": 1, "mu": [[0.0, 0.0]]}],
        # Optional parameter of type Dict[str, float] or None. Defines the parameters
        # of the target function. Leave to None for random sampling according to default
        # probabilities for the respective function family.
        "parameters_target": {"theta": 1, "mu": [[2, 2]]},
        "dir_target_points": "/mnt/users/daijun_chen/gits/github/Shape-Transfer-Bayesian-Optimization/data/EXP_mu2_2_2_theta_1",
        "file_target_points": "simExp_points_task2"
    },
    # Mandatory parameter. Defines the magnitude of the i.i.d. measurement noise.
    "output_noise": 0.0,
    # Mandatory parameter. The start point of transfer GP, [ random | source_best ]
    "start_bo": "source_best"
}

