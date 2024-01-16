import os, gc, scipy.sparse
import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import TruncatedSVD
import config

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-t', '--task', default='cite', type=str)
args = parser.parse_args()

subtask = args.task

