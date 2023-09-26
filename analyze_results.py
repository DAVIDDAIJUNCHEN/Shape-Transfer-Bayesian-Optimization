#!/usr/bin/env python3 
import argparse, os, random
import numpy as np


def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="Analyze results in a dir")
    argparser.add_argument("input_dir", description="input dir containing subdirs of experiments")
    argparser.add_argument("")    
    
