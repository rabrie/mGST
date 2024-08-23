from mGST import compatibility,low_level_jit,additional_fns, algorithm

import csv
import os

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scipy.optimize import linear_sum_assignment
from scipy.linalg import logm
from scipy.optimize import minimize
from argparse import Namespace
from random import sample

import pandas as pd
from pygsti.tools import change_basis
from pygsti.baseobjs import Basis
from pygsti.report.reportables import entanglement_fidelity
from pygsti.algorithms import gaugeopt_to_target
from pygsti.models import gaugegroup
from pygsti.tools.optools import compute_povm_map

from qiskit.quantum_info import SuperOp
from qiskit.quantum_info.operators.measures import diamond_norm



# rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm,amsmath,amssymb,lmodern}')
plt.rcParams.update({'font.family':'computer-modern'})


SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def multikron(matrix_array):
    """ Computes the Kronecker product of all matrices in an array
    The order is matrix_array[0] otimes matrix_array[1] otimes ...

    Parameters
    ----------
    matrix_array: numpy array
        An array containing matrices

    Returns
    -------
    res: numpy array
        The resulting tensor product (potentially a very large array)
    """
    res = matrix_array[0]
    for i in range(1,matrix_array.shape[0]):
        res = np.kron(res, matrix_array[i])
    return res

def min_spectral_distance(X1,X2):
    """ Computes the average absolute distance between the eigenvlues of two matrices
    The matrices are first diagonalized, then the eigenvalues are matched such that the average
    distance of matched eigenvalue pairs is minimal.
    The big advantage of this distance metric is that it is gauge invariant and it can thus be used
    to if the reconstructed gates are similar to the target gates before any gauge optimization.

    Parameters
    ----------
    X1: numpy array
        The first matrix
    X2: numpy array
        The second matrix

    Returns
    -------
    dist: float
        The minimal distance
    """
    r = X1.shape[0]
    eigs = la.eig(X1)[0]
    eigs_t = la.eig(X2)[0]
    cost_matrix = np.array([[np.abs(eigs[i] - eigs_t[j]) for i in range(r)] for j in range(r)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    normalization = np.abs(eigs).sum()
    dist = cost_matrix[row_ind,col_ind].sum()/normalization
    return dist


def MVE_data(X, E, rho, J, y):
    """Mean varation error between measured outcomes and predicted outcomes from a gate set

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence

    Returns
    -------
    dist : float
        Mean variation error
    max_dist : float
        Maximal varaition error

    Notes:
        For each sequence the total variation error of the two probability distribution
        over the POVM elements is computed. Afterwards the meean over these total
        variation errors is returned.
    """
    m = y.shape[1]
    n_povm = y.shape[0]
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C = low_level_jit.contract(X, j)
        curr = 0
        for k in range(n_povm):
            y_model = E[k].conj()@C@rho
            curr += np.abs(y_model - y[k, i])
        curr = curr/2
        dist += curr
        if curr > max_dist:
            max_dist = curr
    return dist/m, max_dist


def gauge_opt(X, E, rho, target_mdl, weights):
    """ Performs pyGSti gauge optimization to target model

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    target_mdl : pygsti model object
        A model containing the target gate set
    weights : dict[str: float]
        A dictionary with keys being gate labels or 'spam' and corresponding values being the
        weight of each gate in the gauge optimization.
        Example for uniform weights: dict({'G%i'%i:1  for i in range(d)}, **{'spam':1})

    Returns
    -------
    X_opt, E_opt, rho_opt: Numpy arrays
        The gauge optimized gates and SPAM arrays
    """
    pdim = int(np.sqrt(rho.shape[0]))
    mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = gaugeopt_to_target(mdl, 
                target_mdl,gauge_group = gaugegroup.UnitaryGaugeGroup(target_mdl.state_space, basis = 'pp'),
                item_weights=weights)
    return compatibility.pygsti_model_to_arrays(gauge_optimized_mdl,basis = 'std')  

def report(X, E, rho, J, y, target_mdl, gate_labels):
    """ Generation of pandas dataframes with gate and SPAM quality measures
    The resutls can be converted to .tex tables or other formats to be used for GST reports

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    target_mdl : pygsti model object
        A model containing the target gate set
    gate_labels : list[str]
        A list of names for the gates in X

    Returns
    -------
        df_g : Pandas DataFrame
            DataFrame of gate quality measures
        df_o : Pandas DataFrame
            DataFrame of all other quality/error measures
        s_g : Pandas DataFrame.style object
        s_o : Pandas DataFrame.style object
    """
    pdim = int(np.sqrt(rho.shape[0]))
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
    E_map = compute_povm_map(gauge_optimized_mdl,'Mdefault')
    E_map_t = compute_povm_map(target_mdl,'Mdefault')
        
    final_objf = low_level_jit.objf(X,E,rho,J,y)
    MVE = MVE_data(X,E,rho,J,y)[0]
    MVE_target = MVE_data(X_t,E_t,rho_t,J,y)[0]
    
    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t))/2
    rho_td = la.norm(rho.reshape((pdim,pdim))-rho_t.reshape((pdim,pdim)),ord = 'nuc')/2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    DD = [diamond_norm(SuperOp(X[i]) - SuperOp(X_t[i]))/2 for i in range(len(X))]
    min_spectral_dists = [min_spectral_distance(X[i],X_t[i]) for i in range(X.shape[0])]


    df_g = pd.DataFrame({
        "F_avg":F_avg,
        "Diamond distances": DD,
        "Min. Spectral distances": min_spectral_dists
    })
    df_o = pd.DataFrame({
        "Final cost function value": final_objf,
        "Mean total variation dist. to data": MVE,
        "Mean total variation dist. target to data": MVE_target,
        "POVM - Meas. map diamond dist.": povm_dd,
        "State - Trace dist.": rho_td,  
    }, index = [0])
    df_g.rename(index=gate_labels, inplace = True)
    df_o.rename(index={0: ""}, inplace = True)
    
    # s_g = df_g.style.format(precision=5, thousands=".", decimal=",")
    # s_o = df_o.style
    #
    # s_g.set_table_styles([
    # {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    # {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    # {'selector': 'td', 'props': 'text-align: center'},
    # ], overwrite=False)
    # s_o.set_table_styles([
    # {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    # {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    # {'selector': 'td', 'props': 'text-align: center'},
    # ], overwrite=False)
    return df_g, df_o#, s_g, s_o

def quick_report(X, E, rho, J, y, target_mdl, gate_labels):
    """ Generation of pandas dataframes with gate and SPAM quality measures
    The quick report is intended to check on a GST estimate with fast to compute measures
    (no diamond distance) to get a first picture and check whether mGST and the gauge optimization
    produce meaningful results.

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    target_mdl : pygsti model object
        A model containing the target gate set
    gate_labels : list[str]
        A list of names for the gates in X

    Returns
    -------
        df_g : Pandas DataFrame
            DataFrame of gate quality measures
        df_o : Pandas DataFrame
            DataFrame of all other quality/error measures
        s_g : Pandas DataFrame.style object
        s_o : Pandas DataFrame.style object
    """
    pdim = int(np.sqrt(rho.shape[0]))
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
        
    final_objf = low_level_jit.objf(X,E,rho,J,y)
    MVE = MVE_data(X,E,rho,J,y)[0]
    MVE_target = MVE_data(X_t,E_t,rho_t,J,y)[0]

    E_map = compute_povm_map(gauge_optimized_mdl,'Mdefault')
    E_map_t = compute_povm_map(target_mdl,'Mdefault')
    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t)) / 2

    rho_td = la.norm(rho.reshape((pdim,pdim))-rho_t.reshape((pdim,pdim)),ord = 'nuc')/2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    min_spectral_dists = [min_spectral_distance(X[i],X_t[i]) for i in range(X.shape[0])]
    

    df_g = pd.DataFrame({
        "F_avg":F_avg,
        "Min. Spectral distances": min_spectral_dists
    })
    df_o = pd.DataFrame({
        "Final cost function": final_objf,
        "Mean TVD estimate-data": MVE,
        "Mean TVD target-data": MVE_target,
        "SPAM error:": rho_td + povm_dd,
    }, index = [0])
    df_g.rename(index=gate_labels, inplace = True)
    df_o.rename(index={0: ""}, inplace = True)
    
    # s_g = df_g.style.format(precision=5, thousands=".", decimal=",")
    # s_o = df_o.style
    #
    # s_g.set_table_styles([
    # {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    # {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    # {'selector': 'td', 'props': 'text-align: center'},
    # ], overwrite=False)
    # s_o.set_table_styles([
    # {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    # {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    # {'selector': 'td', 'props': 'text-align: center'},
    # ], overwrite=False)
    return df_g, df_o #, s_g, s_o


def set_size(w,h, ax=None):
    """ Forcing a figure to a specified size

    Parameters
    ----------
    w : floar
        width in inches
    h : float
        height in inches
    ax : matplotlib axes
        The optional axes
    """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def plot_mat(mat1, mat2):
    """ Quick side by side plot of two matrices with imshow
    Intended to compare GST result and target gate

    Parameters
    ----------
    mat1 : numpy array
        2D numpy array with data
    mat2 : numpy array
        2D numpy array with data
    """
    dim = mat1.shape[0]
    fig, axes = plt.subplots(ncols=2, nrows = 1,gridspec_kw={"width_ratios":[1,1]}, sharex=True)
    plt.rc('image', cmap='RdBu')
    ax = axes[0]
    im0 = ax.imshow(np.real(mat1), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(np.arange(dim)+1)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(np.arange(dim)+1)
    ax.grid(visible = 'True', alpha = 0.4)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax = axes[1]
    im1 = ax.imshow(np.real(mat2), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(np.arange(dim)+1)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(np.arange(dim)+1)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.grid(visible = 'True', alpha = 0.4)
    axes[0].set_title(r'GST result')
    axes[1].set_title(r'Ideal gate')

    # cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0-0.05,0.02,ax.get_position().height])
    #fig.colorbar(im1, cax=cax)
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    cbar.ax.set_ylabel(r'Matrix \, entry $\times 10$', labelpad = 5, rotation=90)


    fig.subplots_adjust(left = 0.05, right = .7, top = 1, bottom = -.1)

    set_size(np.sqrt(3*dim),np.sqrt(dim)*1.2)

    plt.show()
    
def plot_spam(rho, E):
    """ Plots POVM elements and initial state with imshow

    Parameters
    ----------
    E : numpy array
        POVM
    rho : numpy array
        Initial state

    Returns
    -------
    img : Matplotlib image
    """
    r = rho.shape[0]
    n_povm = E.shape[0]
    fig, axes = plt.subplots(ncols = 1, nrows=n_povm+1, sharex=True)
    plt.rc('image', cmap='RdBu')
    
    ax = axes[0]
    im0 = ax.imshow(np.real(rho).reshape(1,r), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(r))
    ax.set_title(r'$\rho$')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    
    for i in range(n_povm): 
        ax = axes[1+i]
        ax.imshow(np.real(E[i].reshape(1,r)), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
        ax.set_xticks(np.arange(r))
        ax.set_xticklabels(np.arange(r)+1)
        ax.set_title(r'POVM-Element %i'%(i+1))
        ax.yaxis.set_major_locator(ticker.NullLocator())

#     cax = fig.add_axes([axes[0].get_position().x1+0.05,ax.get_position().y0-0.05,0.02,10*ax.get_position().height])
#     fig.colorbar(im0, cax=cax)
    
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    cbar.ax.set_ylabel(r'Pauli basis coefficient', labelpad = 5, rotation=90)

    #fig.subplots_adjust(left = 0.05, right = .7, top = 1, bottom = -.1)
    set_size(4,3)
    plt.show()
    return im0


def generate_gate_err_pdf(filename, gates1, gates2, basis_labels = False, gate_labels = False,
                          magnification = 5, return_fig = False):
    """ Main routine to generate plots of reconstructed gates, ideal gates and the noise channels
    of the reconstructed gates.
    The basis is arbitrary but using gates in the Pauli basis is recommended.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    gates1 : numpy array
        A gate set in the same format as the "X"-tensor. These gates are assumed to be the GST estimates.
    gates1 : numpy array
        A gate set in the same format as the "X"-tensor. These are assumed to be the target gates.
    basis_labels : list[str]
        A list of labels for the basis elements. For the standard basis this could be ["00", "01",...]
        and for the Pauli basis ["I", "X", "Y", "Z"] or the multi-qubit version.
    gate_labels : list[str]
        A list of names for the gates
    magnification : float
        A factor to be applied to magnify errors in the rightmost plot.
    return_fig : bool
        If set to True, a figure object is returned by the function, otherwise the plots are saved as <filename>
    """
    d = gates1.shape[0]
    dim = gates1[0].shape[0]
    if not basis_labels:
        basis_labels = np.arange(dim)
    if not gate_labels:
        gate_labels = ['G%i' % k for k in range(d)]
    plot3_title = r'$\mathrm{id} - \hat{\mathcal{G}}\mathcal{U}^{-1}$'
    
    cmap = plt.colormaps.get_cmap('RdBu')
    norm = Normalize(vmin=-1, vmax=1)

    figures = []
    for i in range(d):
        if dim > 16:
            fig, axes = plt.subplots(ncols=1, nrows=3, gridspec_kw={"height_ratios": [1, 1, 1]}, sharex=True)
        else:
            fig, axes = plt.subplots(ncols=3, nrows = 1,gridspec_kw={"width_ratios":[1,1,1]}, sharex=True)
#         plt.rc('image', cmap='RdBu')

        dim = gates1[0].shape[0]
        plot_matrices = [np.real(gates1[i]), np.real(gates2[i]),
                         magnification*(np.eye(dim) - np.real(gates1[i]@la.inv(gates2[i])))]

        for j in range(3):
            ax = axes[j]
            ax.patch.set_facecolor('whitesmoke')
            ax.set_aspect('equal')
            for (x, y), w in np.ndenumerate(plot_matrices[j].T):
                color = 'white' if w > 0 else 'black'
                size = np.sqrt(np.abs(w))
                rect = plt.Rectangle([x + (1-size)/2, y + (1-size)/2], size, size,
                                     facecolor=cmap((w+1)/2), edgecolor=cmap((w+1)/2))
                #print(cmap(size))
                ax.add_patch(rect)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(dim+1), labels = [])
            ax.set_yticks(np.arange(dim+1), labels = [])

            if dim > 16:
                ax.grid(visible='True', alpha=0.4, lw = .1)
                ax.set_xticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels, fontsize=2,  rotation=45)
                ax.set_yticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels, fontsize=2)
            else:
                ax.grid(visible='True', alpha=0.4)
                ax.set_xticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels, rotation=45)
                ax.set_yticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels)

            ax.tick_params(which = 'major', length = 0) #Turn dummy ticks invisible
            ax.tick_params(which = 'minor', top=True, labeltop=True, bottom=False, labelbottom=False, length = 0)
        
        ax.grid(visible = 'True', alpha = 0.4)
        axes[0].set_title(r'$\hat{\mathcal{G}}$', fontsize = 'large')
        axes[0].set_ylabel('Gate: ' + gate_labels[i], rotation = 90, fontsize = 'large')
        axes[1].set_title(r'$\mathcal{U}$', fontsize = 'large')
        axes[2].set_title(plot3_title, fontsize = 'large')
        

        if dim > 16:
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), pad=0, shrink = 0.6)
            fig.subplots_adjust(left = 0.1, right = .76, top = .85, bottom = .03)
            set_size(0.5 * np.sqrt(dim), 1.3 * np.sqrt(dim))
        else:
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), pad=0)
            fig.subplots_adjust(left=0.1, right=.76, top=.85, bottom=.03, hspace = .2)
            set_size(2 * np.sqrt(dim), 0.8 * np.sqrt(dim))
        figures.append(fig)
        if not return_fig:
            plt.savefig(filename + "G%i.pdf" %i, dpi=150, transparent=True, bbox_inches='tight')
            plt.close()
    if return_fig:
        return figures

    
    
def compute_angles_axes(U_set, alternative_phase = False):
    """ Takes the matrix logarithm of the given unitaries and returns the Hamiltonian parameters
    The parametrization is U = exp(-i \pi H/2), i.e. H = i log(U)*2/pi

    Parameters
    ----------
    U_set: list[numpy array]
        A list contining unitary matrices
    alternative_phase: bool
        Whether an attempt should be made to report more intuitive rotations,
        for example a rotation of 3\pi/2 around the -X axis would be turned into
        a pi/2 rotation around the X-axis.
    Returns
    -------
    angles: list[float]
        The rotation angle on the Bloch sphere for all gates
    axes : list[numpy array]
        The normalized rotation axes in the Pauli basis for all gates
    pauli_coeffs : list[numpy array]
        The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes: sqrt(pdim) factor is due to Pauli basis normalization
    """
    d = U_set.shape[0]
    pdim = U_set.shape[1]
    angles = []
    axes = []
    pp_vecs = []
    for i in range(d):
        H = 1j*logm(U_set[i])
        pp_vec = change_basis(H.reshape(-1),'std','pp')
        original_phase = la.norm(pp_vec[1:])*2/np.sqrt(pdim)
        if alternative_phase and (-np.min(pp_vec) > np.max(pp_vec)) and original_phase > np.pi:
            alt_phase = (-original_phase + 2*np.pi)%(2*np.pi)
            pp_vec = - pp_vec
        else:
            alt_phase = original_phase
        angles.append(alt_phase/np.pi) 
        axes.append(pp_vec[1:]/la.norm(pp_vec[1:]))
        pp_vecs.append(pp_vec)
    pauli_coeffs = np.array(pp_vecs)/np.sqrt(pdim)/np.pi*2
    return angles, axes, pauli_coeffs

def compute_sparsest_Pauli_Hamiltonian(U_set):
    """ Takes the matrix logarithms of the given unitaries and returns sparsest Hamiltonian parameters in Pauli basis
    The parametrization is U = exp(-i \pi H/2), i.e. H = i log(U)*2/pi.
    Different branches in the matrix logarithm lead to different Hamiltonians. This function optimizes over
    combinations of adding 2*\pi to different eigenvalues, in order arrive at the branch with the Hamiltonian
    whose Pauli basis representation is the most sparse.


    Parameters
    ----------
    U_set  : list[numpy array]
        A list contining unitary matrices

    Returns
    -------
    pauli_coeffs : list[numpy array]
        The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes: sqrt(pdim) factor is due to Pauli basis normalization
    """
    pdim = U_set.shape[1]
    pp_vecs = []

    for U in U_set:
        evals, evecs = np.linalg.eig(U)
        Pauli_norms = []
        for i in range(2 ** pdim):
            bits = low_level_jit.local_basis(i, 2, pdim)
            evals_new = 1j * np.log(evals) + 2 * np.pi * bits
            H_new = evecs @ np.diag(evals_new) @ evecs.T.conj()
            pp_vec = change_basis(H_new.reshape(-1), 'std', 'pp')
            Pauli_norms.append(np.linalg.norm(pp_vec, ord=1))
        opt_bits = low_level_jit.local_basis(np.argsort(Pauli_norms)[0], 2, pdim)
        evals_opt = 1j * np.log(evals) + 2 * np.pi * opt_bits
        H_opt = evecs @ np.diag(evals_opt) @ evecs.T.conj()
        pp_vecs.append(change_basis(H_opt.reshape(-1), 'std', 'pp'))
    pauli_coeffs = np.array(pp_vecs) / np.sqrt(pdim) / np.pi * 2
    return pauli_coeffs

def read_sequences_from_csv(filename):
    """ Load mGST sequences which were saved into csv format back into a numpy array with correct formating

    Parameters
    ----------
    filename : str
        The file name of the .csv file relative to the current path or as absolute path

    Returns
    -------
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    l_max : int
        The maximum sequence length (also refered to as circuit depth)

    """
    # Import sequence list and load data
    J_list = pd.read_csv(filename, delimiter=",", names=list(range(32))).values
    N = J_list.shape[0] + 1
    J_list = [[int(x) for x in J_list[i, :] if str(x) != 'nan'] for i in range(N - 1)]
    J_list.insert(0, [])
    l_max = np.max([len(J_list[i]) for i in range(N)])

    J = []
    for i in range(N):
        J.append(list(np.pad(J_list[i], (0, l_max - len(J_list[i])), 'constant', constant_values=-1)))
    # Reversing of the circuit order (GST circuits are read from right to left)
    J = np.array(J).astype(int)[:, ::-1]
    return J, l_max


def generate_spam_err_pdf(filename, E, rho, E2, rho2, basis_labels = False, spam2_content = 'ideal'):
    """ Generate pdf plots of two sets of POVM + state side by side in vector shape
    The input sets can be either POVM/state directly or a difference different SPAM parametrizations to
    visualize errors.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    E2 : numpy array
        POVM #2
    rho2 : numpy array
        Initial state #2
    basis_labels : list[str]
        A list of labels for the basis elements. For the standard basis this could be ["00", "01",...]
        and for the Pauli basis ["I", "X", "Y", "Z"] or the multi-qubit version.
    spam2_content : str
        Label of the right SPAM plot to indicate whether it is the ideal SPAM parametrization or for instance
        the error between the reconstructed and target SPAM

    Returns
    -------
    """
    r = rho.shape[0]
    pdim = int(np.sqrt(r))
    n_povm = E.shape[0]
    fig, axes = plt.subplots(ncols = 2, nrows=n_povm+1, sharex=True)
    plt.rc('image', cmap='RdBu')

    ax = axes[0,0]
    im0 = ax.imshow(rho, vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(r))
    ax.set_title(r'$\rho$')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    
    ax = axes[0,1]
    im0 = ax.imshow(rho2, vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(r))
    ax.set_title(r'$\rho$ - ' + spam2_content)
    ax.yaxis.set_major_locator(ticker.NullLocator())

    for i in range(n_povm): 
        ax = axes[1+i,0]
        ax.imshow(E[i], vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
        ax.set_xticks(np.arange(pdim))
        ax.set_xticklabels(np.arange(pdim)+1)
        ax.set_title(r'E%i'%(i+1))
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())

        ax = axes[1+i,1]
        ax.imshow(E2[i], vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
        ax.set_xticks(np.arange(pdim))
        ax.set_xticklabels(np.arange(pdim)+1)
        ax.set_title(r'E%i - '%(i+1) + spam2_content)
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    cbar.ax.set_ylabel(r'Pauli basis coefficient', labelpad = 5, rotation=90)

    fig.subplots_adjust(left = .1, right = .7, top = .95, bottom = .05, hspace = .5)

    set_size(3,2)
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()
    
def generate_spam_err_std_pdf(filename, E, rho, E2, rho2, basis_labels = False, magnification = 10, return_fig = False):
    """ Generate pdf plots of two sets of POVM + state side by side in matrix shape
    The input sets can be either POVM/state directly or a difference different SPAM parametrizations to
    visualize errors.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    E2 : numpy array
        POVM #2
    rho2 : numpy array
        Initial state #2
    basis_labels : list[str]
        A list of labels for the basis elements. For the standard basis this could be ["00", "01",...]
        and for the Pauli basis ["I", "X", "Y", "Z"] or the multi-qubit version.
    magnification : float
        A factor to be applied to magnify errors in the rightmost plot.
    return_fig : bool
        If set to True, a figure object is returned by the function, otherwise the plot is saved as <filename>
    Returns
    -------
    """
    dim = rho.shape[0]
    pdim = int(np.sqrt(dim))
    n_povm = E.shape[0]
    cmap = plt.colormaps.get_cmap('RdBu')
    norm = Normalize(vmin=-1, vmax=1)
        
    fig, axes = plt.subplots(ncols=3, nrows = n_povm + 1,gridspec_kw={"width_ratios":[1,1,1]}, sharex=True)
    plt.rc('image', cmap='RdBu')

    for i in range(n_povm+1):
        if i == 0:
            plot_matrices = [np.real(rho), np.real(rho2), np.real(rho - rho2)]
            axes[i,0].set_ylabel(r'$\hat{\rho}$', rotation = 90, fontsize = 'large')
        else: 
            plot_matrices = [np.real(E[i-1]), np.real(E2[i-1]), np.real(E[i-1] - E2[i-1])*magnification]
            # axes[i,0].set_title(r'$\hat{E}$', fontsize = 'large')
            axes[i,0].set_ylabel(r'$\hat{E}_%i$'%(i-1), rotation = 90, fontsize = 'large')
            # axes[i,1].set_title(r'$E_{\mathrm{ideal}}$', fontsize = 'large')
            # axes[i,2].set_title(r'$E - E_{\mathrm{ideal}}$', fontsize = 'large')
            
        for j in range(3):
            ax = axes[i,j]
            ax.patch.set_facecolor('whitesmoke')
            ax.set_aspect('equal')
            for (x, y), w in np.ndenumerate(plot_matrices[j].reshape(pdim,pdim)):
                color = 'white' if w > 0 else 'black'
                size = np.sqrt(np.abs(w))
                rect = plt.Rectangle([x + (1-size)/2, y + (1-size)/2], size, size,
                                     facecolor=cmap((w+1)/2), edgecolor=cmap((w+1)/2))
                #print(cmap(size))
                ax.add_patch(rect)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(pdim+1), labels = [])

            ax.set_yticks(np.arange(pdim+1), labels = [])
            #ax.set_yticklabels(basis_labels)
            ax.tick_params(which = 'major', length = 0) #Turn dummy ticks invisible
            ax.tick_params(which = 'minor', top=True, labeltop=True, bottom=False, labelbottom=False, length = 0, pad = 1)

            if pdim > 4:
                ax.grid(visible='True', alpha=0.4, lw=.1)
                ax.set_xticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, rotation=45, fontsize=2)
                ax.set_yticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, fontsize=2)
            else:
                ax.grid(visible='True', alpha=0.4)
                ax.set_xticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, rotation=45, fontsize=6)
                ax.set_yticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, fontsize=6)

    if dim > 16:
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0, shrink = 0.6)
        fig.subplots_adjust(left=0, right=.7, top=.90, bottom=.05, wspace=-.6, hspace=0.4)
    else:
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0)
        fig.subplots_adjust(left=0, right=.7, top=.90, bottom=.05, wspace=-.6, hspace=0.8)

    # cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    #cbar.ax.set_ylabel(r'Pauli basis coefficient', labelpad = 5, rotation=90)

    # set_size(3,2)
    #set_size(2*np.sqrt(pdim),0.8*np.sqrt(n_povm+1))
    if return_fig:
        return fig
    else:
        plt.savefig(filename, dpi=150, transparent=True)
        plt.close()
    
    
def phase_err(angle, U, U_t):
    """ Computes norm between two input unitaries after a global phase is added to one of them

    Parameters
    ----------
    angle : float
        The global phase angle (in rad)
    U : numpy array
        The first unitary matrix
    U_t : numpy array
        The second unitary matrix (typically the a target gate)

    Returns
    -------
    norm : floar
        The norm of the difference
    """
    return la.norm(np.exp(1j*angle)*U - U_t)

def phase_opt(X, K_t):
    """ Return rK = 1 gate set with global phase fitting matching to target gate set

    Parameters
    ----------
    X: 3D numpy array
        Array where CPT superoperators are stacked along the first axis.
        These should correspond to rK = 1 gates for the outputs to be meaningful.
    K_t: 4D numpy array
        Array of target gate Kraus operators
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Returns
    -------
    K_opt: 4D numpy array
        Array of Kraus operators with mathed global phase
    """
    d = X.shape[0]
    r = X.shape[1]
    pdim = int(np.sqrt(r))
    K = additional_fns.Kraus_rep(X,d,pdim,1).reshape(d,pdim,pdim)
    K_t = K_t.reshape(d,pdim,pdim)
    K_opt = np.zeros(K.shape).astype(complex)
    for i in range(d):
        angle_opt = minimize(phase_err, 1e-9, args=(K[i], K_t[i]), method='COBYLA').x
        K_opt[i] = K[i]*np.exp(1j*angle_opt)
    return K_opt

def eff_depol_params(X_opt_pp):
    """ Computes the entanglement fidelities to the completely depolarizing channel

    Parameters
    ----------
    X_opt_pp: 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    ent_fids : list[float]
        List of entanglement fidelities corresponding to the gates in X_opt_pp.
    """
    r = X_opt_pp.shape[1]
    ent_fids = []
    basis = Basis.cast('pp', r)
    K_depol = additional_fns.depol(int(np.sqrt(r)),1)
    X_depol = np.einsum("jkl,jnm -> knlm", K_depol, K_depol.conj()).reshape(r, r)
    for i in range(X_opt_pp.shape[0]):
        ent_fids.append(entanglement_fidelity(X_opt_pp[i], change_basis(X_depol, 'std', 'pp'), basis))
    return ent_fids

def eff_depol_params_agf(X_opt_pp):
    """ Computes the average gate fidelities to the completely depolarizing channel

    Parameters
    ----------
    X_opt_pp: 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    ent_fids : list[float]
        List of average gate fidelities wrt. the depol. channel corresponding to the gates in X_opt_pp.
    """
    r = X_opt_pp.shape[1]
    pdim = np.sqrt(r)
    ent_fids = []
    basis = Basis.cast('pp', r)
    K_depol = additional_fns.depol(int(np.sqrt(r)),1)
    X_depol = np.einsum("jkl,jnm -> knlm", K_depol, K_depol.conj()).reshape(r, r)
    for i in range(X_opt_pp.shape[0]):
        ent_fids.append(entanglement_fidelity(X_opt_pp[i], change_basis(X_depol, 'std', 'pp'), basis))
    return (pdim*np.array(ent_fids) + 1)/(pdim + 1)

def unitarities(X_opt_pp):
    """ Computes the unitarities of all gates in the gate set

    Parameters
    ----------
    X_opt_pp : 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    unitarities : list[float]
        List of unitarities for the gates in X_opt_pp.
    """
    #Definition: Proposition 1 of https://arxiv.org/pdf/1503.07865.pdf
    pdim = int(np.sqrt(X_opt_pp.shape[1]))
    E_u = X_opt_pp[:,1:,1:] # unital submatrices
    unitarities = np.real(np.einsum('ijk, ijk -> i', E_u.conj(), E_u)/(pdim**2-1))
    return unitarities

# Spectrum of the Choi matrix
def generate_Choi_EV_table(X_opt, n_evals, gate_labels, filename = None):
    """ Outputs a .tex document containing a table with the larges eigenvalues of the Choi matrix for each gate

    Parameters
    ----------
    X_opt : 3D numpy array
        Array where CPT superoperators in standard basis are stacked along the first axis.
    n_evals : int
        Number of eigenvalues to be returned
    gate_labels : list[int: str]
        The names of gates in the gate set
    filename :
        The file name of the output .tex file
    """
    d,r,_ = X_opt.shape
    pdim = int(np.sqrt(r))
    Choi_evals_result = np.zeros((d,r))
    X_choi = X_opt.reshape(d,pdim,pdim,pdim,pdim)
    X_choi = np.einsum('ijklm->iljmk',X_choi).reshape(d,pdim**2,pdim**2)
    for j in range(d):
        Choi_evals_result[j,:] =  np.sort(np.abs(la.eig(X_choi[j])[0]))[::-1]
    Choi_evals_normalized = np.einsum('ij,i -> ij', Choi_evals_result,1/la.norm(Choi_evals_result, axis = 1, ord = 1))

    #Choi_evals_normalized_str = [[number_to_str(Choi_evals_normalized[i,j], precision = 5) for i in range(d)] for j in range(n_evals)]
    df_g_evals = pd.DataFrame(Choi_evals_normalized)
    df_g_evals.rename(index=gate_labels, inplace = True)
    if filename:
        df_g_evals.style.to_latex(filename+'.tex', column_format = 'c|*{%i}{c}'%n_evals, position_float = 'centering',
                               hrules = True, caption = 'Eigenvalues of the Choi state', position = 'h!')
    return df_g_evals


    
    
def local_dephasing_pp(prob_vec):
    """ Returns the tensor product of single qubit dephasing channels in Pauli basis

    Parameters
    ----------
    prob_vec : list[float]
        A list of dephasing probabilities

    Returns
    -------
    D_final : numpy array
        Process matrix of the tensor product of local dephasing channels

    """
    D_loc_array = np.array([[[1,0,0,0],[0,1-2*p,0,0],[0,0,1-2*p,0],[0,0,0,1]] for p in prob_vec])
    D_final = multikron(D_loc_array)
    return D_final

def dephasing_dist(prob_vec, X_pp):
    """ Returns the distance between a given channel and a local dephasing channel with given probabilities

    Parameters
    ----------
    prob_vec : list[float]
        A list of dephasing probabilities
    X_pp : 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    norm : float
        The norm between the channel difference

    """
    X_deph = local_dephasing_pp(prob_vec)
    return la.norm(X_pp-X_deph)

def dephasing_probabilities_2q(X_opt_pp, X_ideal_pp):
    """ Determines the local dephasing channel parameters which best describe the noise model
    Works for two qubit gates only

    Parameters
    ----------
    X_opt_pp : 3D numpy array
        Array where reconstructed CPT superoperators in Pauli basis are stacked along the first axis.
    X_opt_pp : 3D numpy array
        Array where target gate CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    dephasing_probs : list[float]
        The two best fit dephasing probabilities
    """
    dephasing_probs = []
    for i in range(X_opt_pp.shape[0]):
        dephasing_probs.append(minimize(dephasing_dist, [0.1,0.1], args = (X_opt_pp[i]@la.inv(X_ideal_pp[i]))).x)
    return dephasing_probs
        

def bootstrap_errors(K, X, E, rho, mGST_args, bootstrap_samples, weights, gate_labels, target_mdl, parametric = False):
    """ Resamples circuit outcomes a number of times and computes GST estimates for each repetition
    All results are then returned in order to compute bootstrap-error bars for GST estimates.
    Parametric bootstrapping uses the estimated gate set to create a newly sampled data set.
    Non-parametric bootstrapping uses the initial dataset and resamples according to the
    corresp. outcome probabilities.
    Each bootstrap run is initialized with the estimated gate set in order to save processing time.

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    X : 3D numpy array
        Array where reconstructed CPT superoperators in standard basis are stacked along the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    mGST_args : dict[str: misc]
        Arguments with which the run_mGST function was called
    bootstrap_samples : int
        Number of bootstrapping repretitions
    weights : dict[str: float]
        Gate weights used for gauge optimization
    gate_labels : list[int: str]
        The names of gates in the gate set
    target_mdl : pygsti model object
        The target gate set
    parametric : bool
        If set to True, parametric bootstrapping is used, else non-parametric bootstrapping. Default: False

    Returns
    -------
    X_array : numpy array
        Array containing all estimated gate tensors of different bootstrapping repretitions along first axis
    E_array : numpy array
        Array containing all estimated POVM tensors of different bootstrapping repretitions along first axis
    rho_array : numpy array
        Array containing all estimated initial states of different bootstrapping repretitions along first axis
    df_g_array : numpy array
        Contains gate quality measures of bootstrapping repetitions
    df_o_array : numpy array
        Contains SPAM and other quality measures of bootstrapping repetitions

    """
    ns = Namespace(**mGST_args)
    if parametric:
        y = np.real(np.array([[E[i].conj()@low_level_jit.contract(X,j)@rho for j in ns.J] for i in range(ns.n_povm)]))
    else: 
        y = ns.y
    X_array = np.zeros((bootstrap_samples, *X.shape)).astype(complex)
    E_array = np.zeros((bootstrap_samples, *E.shape)).astype(complex)
    rho_array = np.zeros((bootstrap_samples, *rho.shape)).astype(complex)
    df_g_list = []
    df_o_list = []
    
    for i in range(bootstrap_samples):        
        y_sampled = additional_fns.sampled_measurements(y, ns.meas_samples).copy()
        K_, X_, E_, rho_, _ = algorithm.run_mGST(y_sampled,ns.J,ns.l,ns.d,ns.r,ns.rK, ns.n_povm, ns.bsize, ns.meas_samples, method = ns.method,
                         max_inits = ns.max_inits, max_iter = 0, final_iter = ns.final_iter, threshold_multiplier = ns.threshold_multiplier, 
                         target_rel_prec = ns.target_rel_prec, init = [K, E, rho], testing = False)   

        X_opt, E_opt, rho_opt = gauge_opt(X_, E_, rho_, target_mdl, weights)
        df_g, df_o = report(X_opt, E_opt, rho_opt, ns.J, y_sampled, target_mdl, gate_labels)
        df_g_list.append(df_g.values)
        df_o_list.append(df_o.values)

        X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)
        
        X_array[i, :] = X_opt_pp
        E_array[i, :] = E_opt_pp
        rho_array[i, :] = rho_opt_pp
        
    return (X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list))

def job_counts_to_mGST_format(self, result_dict):
    """ Turns the dictionary of outcomes obtained from qiskit backend
        into the format which is used in mGST

    Parameters
    ----------
    result_dict: (dict of str: int)

    Returns
    -------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence

    """
    basis_dict_list = []
    for result in result_dict:
        # Translate dictionary entries of bitstring on the full system to the decimal representation of bitstrings on the active qubits
        basis_dict = {entry: int("".join([entry[::-1][i] for i in self.qubits][::-1]), 2) for entry in result}
        # Sort by index:
        basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))
        basis_dict_list.append(basis_dict)
    y = []
    for i in range(len(result_dict)):
        row = [result_dict[i][key] for key in basis_dict_list[i]]
        if len(row) < self.num_povm:
            missing_entries = list(np.arange(self.num_povm))
            for given_entry in basis_dict_list[i].values():
                missing_entries.remove(given_entry)
            for missing_entry in missing_entries:
                row.insert(missing_entry, 0)  # 0 measurement outcomes in not recorded entry
        y.append(row / np.sum(row))
    y = np.array(y).T
    return y

def outcome_probs_from_files(folder_name, basis_dict, n_povm,N):
    """ Searches a specified folder for .txt files containing circuit outcomes and combines the results
    Each text file needs to have line in the following format:
    1: 1,0,1,0,1,1,0,0,0
    2: 1,1,1,1,0,1,1,1,1
    Here the first number specifies the circuit and basis outcome of each shot is written to the right.

    Parameters
    ----------
    folder_name : str
        The relative or absolute name of the data folder
    basis_dict : dict[str: int]
        Translation between label for each shot in the .txt files and the numbering of the POVM element.
        Example (two qubits): {'00': 0, '01': 1, '10': 2, '11': 3,}
    n_povm : int
        The number of POVM elements in the data set
    N : int
        The number of circuits in the data set

    Returns
    -------
    y : numpy array
        2D array of measurement outcomes for all measured sequences;
        Each column contains the outcome probabilities for a fixed sequence
    avg_counts : int
        The average number of shots per circuit after combining data sets
    """
    filenames = os.listdir(path = folder_name)
    datafile_names  = [s for s in filenames if ".txt" in s]

    # sample_counts keeps track of how many shots were taken for each sequence
    sample_counts = []
    # array of outcome probabilities:
    y = np.zeros((n_povm,N))
    sample_counts = np.zeros((len(datafile_names),N))
    k = 0
    for filename in datafile_names:
        with open(folder_name + "/" + filename) as file:
            i = 0
            for line in file:
                # removing row index at beginning and \n mark at end 
                line_entries = line.rstrip().split(': ')
                # splitting result string at commas
                result_list = line_entries[1:][0].split(',')
                # translate each measurement result onto basis index
                for entry in result_list: 
                    j = basis_dict[entry]
                    y[j,i] += 1
                sample_counts[k,i] = len(result_list)
                i += 1
        k += 1
    total_counts = np.sum(sample_counts, axis = 0)
    avg_counts = int(np.average(total_counts))
    return y/total_counts, avg_counts

def save_var_latex(key, value):
    """ Saves variables in data file to be read in latex document
    Credit to https://stackoverflow.com/a/66620671

    Parameters
    ----------
    key : str
        The name of the variable
    value : misc
        The variable content, could be a string as in the name of the experiment, or the number of circuits run (int),
        or any other python variable that needs to be transfered
    """

    dict_var = {}

    file_path = os.path.join(os.getcwd(), "report/latex_vars.dat")

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")
            

def n_params(pdim,d,rK,n_povm):
    """ Returns the number of free parameters in a gate set
    Unitary gauge parameters and unitary freedom in representing the Kraus operators and the POVM elements are
    substracted. For details, see
    https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-71735/dissertation_brieger-2%281%29.pdf

    Parameters
    ----------
    pdim : int
        The physical dimension, i.e. 2^(#qubits)
    d : int
        The number of gates in the gate set
    rK : int
        The Kraus rank of the gates
    n_povm : int
        The number of POVM elements; For computational basis measurements n_povm = pdim

    Returns
    -------
    n_params : int
        The number of free parameters
    """
    # Order: gates + stat + povm - povm_freedom - gauge freedom - Kraus freedom
    return d*(pdim**2*(2*rK - 1) - rK**2) + pdim*(n_povm*pdim - pdim)

def number_to_str(number, uncertainty = None, precision = 3):
    """
    Formats a floating point number to a string with the given precision.

    Parameters:
    number (float): The floating point number to format.
    uncertainty (tuple): The upper and lower values of the confidence interval
    precision (int): The number of decimal places to include in the formatted string.

    Returns:
    str: The formatted floating point number as a string.
    """
    if uncertainty is None:
        return f"{number:.{precision}f}"

    return f"{number:.{precision}f} [{uncertainty[1]:.{precision}f},{uncertainty[0]:.{precision}f}]"