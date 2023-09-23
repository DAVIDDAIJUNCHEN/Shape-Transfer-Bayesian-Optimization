#!/usr/bin/env python3 

import argparse, os, random
import numpy as np
from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO


def arg_parser():
    "parse the argument"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")
    argparser.add_argument("--theta", default="1.0", help="parameter in exponential target function")
    argparser.add_argument("--mu1", default="[0.0, 0.0]", help="mu of target function 1")
    argparser.add_argument("--mu2", default="[0.5, 0.5]", help="mu of target function 2")
    argparser.add_argument("--T1", default=10, help="number of experiments in target function 1")
    argparser.add_argument("--T2", default=4, help="number of experiemnts in target function 2")
    argparser.add_argument("--type", default="EXP", choices=["EXP", "BR"], help="choose target function type")

    parser = argparser.parse_args()
    
    return parser

def main_exp():
    "main function of exponential target function type"
    theta = parser.theta
    mu1 = parser.mu1
    mu2 = parser.mu2

    print(theta)
    print(mu1)
    print(mu2)


def main_br():
    "main function of Brainn target function type"
    pass



if __name__ == "__main__":
    parser = arg_parser()

    fun_type = parser.type
    T1 = parser.T1
    T2 = parser.T2

    if fun_type == "EXP":
        mu_1 = parser.mu1
        mu_2 = parser.mu2

        main_exp()
    elif fun_type == "BR":
        main_br()

