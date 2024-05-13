#!/usr/bin/env python
# coding: utf-8

# # Single qubit GST (Data from 2.6.2023)

import sys,os

from mGST import compatibility,algorithm, optimization, low_level_jit, additional_fns
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
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pygsti.tools import change_basis
from matplotlib import rcParams
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from uncertainty import number_to_str
import reporting

# rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm,amsmath,amssymb,lmodern,dsfont}')
plt.rcParams.update({'font.family':'computer-modern'})


SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#### Command line arguments
filename = sys.argv[1]
folder_name = sys.argv[2]
    
# ### Loading the experiment parameters

with open(filename, 'rb') as handle:
    b = pickle.load(handle)
results = b['results']
parameters = b['parameters']

d = parameters['d']
r = parameters['r']
pdim = int(np.sqrt(r))
rK = parameters['rK']
n_povm = parameters['n_povm']
meas_samples = parameters['meas_samples']
J = parameters['J']
y = parameters['y']


# ### Target gate set

S3 = np.array([[1,0],[0,1j]])
S3_2 = np.kron(S3,S3)
X_S3_2 = np.kron(S3_2, S3_2.conj())

from true_values import E_true, X_true, rho_true
X_ideal = X_true[:d].copy()
X_ideal[5] = X_S3_2@X_ideal[5] # Target is CZ with additional local phase gates
E_ideal = E_true.copy()
rho_ideal = rho_true.copy()
mdl_ideal = compatibility.arrays_to_pygsti_model(X_ideal,E_ideal,rho_ideal, basis = 'std')
K_ideal = additional_fns.Kraus_rep(X_ideal,d,pdim,1)


# ### Generating gate measure table

weights = {'G0': 1,'G1': 1, 'G2': 1, 'G3': 1, 'G4': 1, 'G5': 1, 'spam': 1}
gate_labels = {0: "Idle", 1: "Rx(pi/2):0", 2: "Ry(pi/2):0", 3: "Rx(pi/2):1", 4: "Ry(pi/2):1", 5: r"$\mathrm{e}^{-\mathrm{i} \frac{\pi}{4} Z \otimes Z}$"}
labels_local = [['I','X','Y','Z'], ['I','X','Y','Z']]
separator = ''
Pauli_labels = [separator.join(map(str,x)) for x in product(*labels_local)]
std_labels = ['00', '01', '10', '11']

K,X,E,rho = b['results']['estimates']
X_opt, E_opt, rho_opt = b['results']['gauge_opt_estimates']
X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)
X_ideal_pp, E_ideal_pp, rho_ideal_pp = compatibility.std2pp(X_ideal, E_ideal, rho_ideal)


mGST_args = b['parameters']
K,X,E,rho = b['results']['estimates']
J = b['parameters']['J']
y = b['parameters']['y']

df_g, df_o, s_g, s_o = reporting.report(X_opt, E_opt, rho_opt, J, y, mdl_ideal, gate_labels)


df_g_err = pd.DataFrame({
    "Average gate Fidelity": [number_to_str(df_g.values[i,0], precision = 4) for i in range(len(gate_labels))],
    "Diamond distance": [number_to_str(df_g.values[i,1],  precision = 4) for i in range(len(gate_labels))],
    "Unitarity": [number_to_str(reporting.unitarities(X_opt_pp)[i],  precision = 4) for i in range(len(gate_labels))],
    "Entanglemen fidelity to depol. channel": [number_to_str(reporting.eff_depol_params(X_opt_pp)[i],  precision = 4) 
                                               for i in range(len(gate_labels))],
    # "Min. spectral distances": [number_to_str(df_g.values[i,2], precision = 4) for i in range(len(gate_labels))]
})
df_o_err = pd.DataFrame({
    "Final cost": number_to_str(df_o.values[0,0], precision = 4),
    "Mean TVD: estimate - data": number_to_str(df_o.values[0,1],  precision = 4),
    "Mean TVD: target - data": number_to_str(df_o.values[0,2],  precision = 4),
    "POVM - diamond dist.": number_to_str(df_o.values[0,3],  precision = 4),
    "State - trace dist.": number_to_str(df_o.values[0,4],  precision = 4),  
}, index = [0])
df_g_err.rename(index=gate_labels, inplace = True)
df_o_err.rename(index={0: ""}, inplace = True)



df_g_err.style.to_latex(folder_name + '/gate_errors.tex', column_format = 'c|c|c|c|c|c', position_float = 'centering', 
                        hrules = True, caption = 'Gate quality measures', position = 'h!')
df_o_err.style.to_latex(folder_name + '/spam_errors.tex', column_format = 'c|c|c|c|c|c', position_float = 'centering', 
                        hrules = True, caption = 'State and measurement quality measures', position = 'h!')

# ## Spectrum of the Choi matrix
reporting.generate_Choi_EV_table(X_opt, 7, gate_labels, folder_name + '/Choi_evals')



# ## Pauli basis

gates = X_opt_pp
gates_ideal = X_ideal_pp
filename = folder_name + '/pp'
reporting.generate_gate_err_pdf(filename, gates, gates_ideal, basis_labels = Pauli_labels)


# ## SPAM errors

# filename = folder_name + '/spam_errs_pp.pdf'
# reporting.generate_spam_err_pdf(filename, E_opt_pp.real.reshape(n_povm,1,r), rho_opt_pp.real.reshape(1,r), 
#                                 E_ideal_pp.real.reshape(n_povm,1,r), rho_ideal_pp.real.reshape(1,r),  basis_labels = Pauli_labels, spam2_content = 'ideal')

filename = folder_name + '/spam_errs_std_real.pdf'
reporting.generate_spam_err_std_pdf(filename, E_opt.real, rho_opt.real,
                                E_ideal.real, rho_ideal.real,  basis_labels = Pauli_labels, spam2_content = 'ideal')

filename = folder_name + '/spam_errs_std_imag.pdf'
reporting.generate_spam_err_std_pdf(filename, E_opt.imag, rho_opt.imag,
                                E_ideal.imag, rho_ideal.imag,  basis_labels = Pauli_labels, spam2_content = 'ideal')
