from mGST import compatibility,algorithm, optimization, low_level_jit, additional_fns
from reporting import uncertainty, reporting

import pickle as pickle
from pygsti.report import reportables as rptbl #Needs cvxpy!
from pygsti.modelpacks import smq1Q_XYI as std
import pygsti
from argparse import Namespace
from itertools import product

import numpy as np
import pandas as pd
import numpy.linalg as la
import json
import pandas
import matplotlib.pyplot as plt
from os import listdir


# Import sequence list
J_list = pandas.read_csv("sequences.csv", delimiter=",", names = list(range(32))).values
J_list = [[int(x) for x in J_list[i,:] if str(x) != 'nan'] for i in range(N-1)]
J_list.insert(0,[])
l_max = np.max([len(J_list[i]) for i in range(N)])

J = []
for i in range(N):
    J.append(list(np.pad(J_list[i],(0,l_max-len(J_list[i])),'constant',constant_values=-1)))
J = np.array(J).astype(int)[:,::-1]