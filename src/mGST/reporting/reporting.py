from mGST import compatibility,low_level_jit,additional_fns, algorithm
from mGST.reporting.uncertainty import *

from pygsti.algorithms import gaugeopt_to_target
from pygsti.models import gaugegroup
from pygsti.tools.optools import compute_povm_map

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
from scipy.linalg import logm
from pygsti.tools import change_basis
from pygsti.baseobjs import Basis
from pygsti.tools.optools import diamonddist
from pygsti.report.reportables import entanglement_fidelity
import csv
import os



from scipy.optimize import minimize
from matplotlib.colors import Normalize
from argparse import Namespace




from matplotlib import rcParams
import matplotlib.ticker as ticker


# rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm,amsmath,amssymb,lmodern}')
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

def multikron(matrix_array):
    res = matrix_array[0]
    for i in range(1,matrix_array.shape[0]):
        res = np.kron(res, matrix_array[i])
    return res

def min_spectral_distance(X1,X2):
    r = X1.shape[0]
    eigs = la.eig(X1)[0]
    eigs_t = la.eig(X2)[0]
    cost_matrix = np.array([[np.abs(eigs[i] - eigs_t[j]) for i in range(r)] for j in range(r)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    normalization = np.abs(eigs).sum()
    return cost_matrix[row_ind,col_ind].sum()/normalization


def MVE_data(X, E, rho, J, y):
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
    pdim = int(np.sqrt(rho.shape[0]))
    mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = gaugeopt_to_target(mdl, 
                target_mdl,gauge_group = gaugegroup.UnitaryGaugeGroup(target_mdl.state_space, basis = 'pp'),
                item_weights=weights)
    return compatibility.pygsti_model_to_arrays(gauge_optimized_mdl,basis = 'std')  

def report(X, E, rho, J, y, target_mdl, gate_labels):
    pdim = int(np.sqrt(rho.shape[0]))
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
    E_map = compute_povm_map(gauge_optimized_mdl,'Mdefault')
    E_map_t = compute_povm_map(target_mdl,'Mdefault')
        
    final_objf = low_level_jit.objf(X,E,rho,J,y)
    MVE = MVE_data(X,E,rho,J,y)[0]
    MVE_target = MVE_data(X_t,E_t,rho_t,J,y)[0]
    
    povm_dd = float(diamonddist(E_map, E_map_t, 'std'))
    rho_td = la.norm(rho.reshape((pdim,pdim))-rho_t.reshape((pdim,pdim)),ord = 'nuc')/2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    DD = compatibility.diamond_dists(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
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
    
    s_g = df_g.style.format(precision=5, thousands=".", decimal=",")
    s_o = df_o.style
    
    s_g.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    s_o.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    return df_g, df_o, s_g, s_o

def quick_report(X, E, rho, J, y, target_mdl, gate_labels):
    pdim = int(np.sqrt(rho.shape[0]))
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
        
    final_objf = low_level_jit.objf(X,E,rho,J,y)
    MVE = MVE_data(X,E,rho,J,y)[0]
    MVE_target = MVE_data(X_t,E_t,rho_t,J,y)[0]
    
    rho_td = la.norm(rho.reshape((pdim,pdim))-rho_t.reshape((pdim,pdim)),ord = 'nuc')/2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    min_spectral_dists = [min_spectral_distance(X[i],X_t[i]) for i in range(X.shape[0])]
    

    df_g = pd.DataFrame({
        "F_avg":F_avg,
        "Min. Spectral distances": min_spectral_dists
    })
    df_o = pd.DataFrame({
        "Final cost function value": final_objf,
        "Mean total variation dist. to data": MVE,
        "Mean total variation dist. target to data": MVE_target,
        "State - Trace dist.": rho_td,  
    }, index = [0])
    df_g.rename(index=gate_labels, inplace = True)
    df_o.rename(index={0: ""}, inplace = True)
    
    s_g = df_g.style.format(precision=5, thousands=".", decimal=",")
    s_o = df_o.style
    
    s_g.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    s_o.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    return df_g, df_o, s_g, s_o


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def plot_mat(mat1, mat2):
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


def generate_gate_err_pdf(filename, gates1, gates2, basis_labels = False, gate_labels = False, magnification = 5):
    d = gates1.shape[0]
    dim = gates1[0].shape[0]
    if not basis_labels:
        basis_labels = np.arange(dim)
    if not gate_labels:
        gate_labels = ['G%i' % k for k in range(d)]
    plot3_title = r'$\hat{\mathcal{G}}\mathcal{U}^{-1}$'
    
    cmap = plt.colormaps.get_cmap('RdBu')
    norm = Normalize(vmin=-1, vmax=1)
    
    for i in range(d):
        fig, axes = plt.subplots(ncols=3, nrows = 1,gridspec_kw={"width_ratios":[1,1,1]}, sharex=True)
#         plt.rc('image', cmap='RdBu')

        dim = gates1[0].shape[0]
        plot_matrices = [np.real(gates1[i]), np.real(gates2[i]), np.real(gates1[i]@la.inv(gates2[i]))]
        
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
            ax.set_xticks(np.arange(dim)+0.5,  minor=True, labels = basis_labels, rotation = 45)

            ax.set_yticks(np.arange(dim+1), labels = [])
            ax.set_yticks(np.arange(dim)+0.5,  minor=True, labels = basis_labels)
            #ax.set_yticklabels(basis_labels)
            ax.grid(visible = 'True', alpha = 0.4)
            ax.tick_params(which = 'major', length = 0) #Turn dummy ticks invisible
            ax.tick_params(which = 'minor', top=True, labeltop=True, bottom=False, labelbottom=False, length = 0)
        
        ax.grid(visible = 'True', alpha = 0.4)
        axes[0].set_title(r'$\hat{\mathcal{G}}$', fontsize = 'large')
        axes[0].set_ylabel('Gate: ' + gate_labels[i], rotation = 90, fontsize = 'large')
        axes[1].set_title(r'$\mathcal{U}$', fontsize = 'large')
        axes[2].set_title(plot3_title, fontsize = 'large')
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm = norm, cmap=cmap), ax=axes.tolist(), pad = 0)

        fig.subplots_adjust(left = 0.1, right = .76, top = .85, bottom = .03)

        set_size(2*np.sqrt(dim),0.8*np.sqrt(dim))
#         plt.show()
        plt.savefig(filename + "G%i.pdf" %i, dpi=150, transparent=True, bbox_inches='tight')
        plt.close()    
    
    
def compute_angles_axes(U_set):
    #Notes: sqrt(pdim) factor is due to Pauli basis normalization
    d = U_set.shape[0]
    pdim = U_set.shape[1]
    angles = []
    axes = []
    pp_vecs = []
    for i in range(d):
        H = 1j*logm(U_set[i])
        pp_vec = change_basis(H.reshape(-1),'std','pp')[1:]
        original_phase = la.norm(pp_vec)*2/np.sqrt(pdim)
        # if (-np.min(pp_vec) > np.max(pp_vec)) and original_phase > np.pi:
        #     alt_phase = (-original_phase + 2*np.pi)%(2*np.pi)
        #     pp_vec = - pp_vec
        # else:
        #     alt_phase = original_phase
        if original_phase > np.pi:
            alt_phase = (-original_phase + 2*np.pi)%(2*np.pi)
            pp_vec = - pp_vec
        else:
            alt_phase = original_phase
        angles.append(alt_phase/np.pi) 
        # print('i:%i'%i, original_phase/np.pi, alt_phase/np.pi)
        axes.append(pp_vec/la.norm(pp_vec))
        pp_vecs.append(pp_vec)
    return angles, axes, np.array(pp_vecs)/np.sqrt(pdim)/np.pi*2


def generate_spam_err_pdf(filename, E, rho, E2, rho2, basis_labels = False, spam2_content = 'ideal'):
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
        
    for i in range(n_povm): 
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
    
def generate_spam_err_std_pdf(filename, E, rho, E2, rho2, basis_labels = False, magnification = 10):
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
            ax.set_xticks(np.arange(pdim)+0.5,  minor=True, labels = basis_labels, rotation = 45, fontsize = 6)

            ax.set_yticks(np.arange(pdim+1), labels = [])
            ax.set_yticks(np.arange(pdim)+0.5,  minor=True, labels = basis_labels, fontsize = 6)
            #ax.set_yticklabels(basis_labels)
            ax.grid(visible = 'True', alpha = 0.4)
            ax.tick_params(which = 'major', length = 0) #Turn dummy ticks invisible
            ax.tick_params(which = 'minor', top=True, labeltop=True, bottom=False, labelbottom=False, length = 0, pad = 1)
            ax.grid(visible = 'True', alpha = 0.4)
        
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm = norm, cmap=cmap), ax=axes, pad = 0)

    fig.subplots_adjust(left = 0.1, right = .76, top = .85, bottom = .03)

    # cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    #cbar.ax.set_ylabel(r'Pauli basis coefficient', labelpad = 5, rotation=90)

    fig.subplots_adjust(left = 0, right = .7, top = .90, bottom = .05, wspace = -.6, hspace = 0.8)

    # set_size(3,2)
    #set_size(2*np.sqrt(pdim),0.8*np.sqrt(n_povm+1))
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()    
    
    
def phase_err(angle, U, U_t):
    return la.norm(np.exp(1j*angle)*U - U_t)

def phase_opt(X, K_t):
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
    r = X_opt_pp.shape[1]
    ent_fids = []
    basis = Basis.cast('pp', r)
    K_depol = additional_fns.depol(int(np.sqrt(r)),1)
    X_depol = np.einsum("jkl,jnm -> knlm", K_depol, K_depol.conj()).reshape(r, r)
    for i in range(X_opt_pp.shape[0]):
        ent_fids.append(entanglement_fidelity(X_opt_pp[i], change_basis(X_depol, 'std', 'pp'), basis))
    return ent_fids

def eff_depol_params_agf(X_opt_pp):
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
    #Definition: Proposition 1 of https://arxiv.org/pdf/1503.07865.pdf
    pdim = int(np.sqrt(X_opt_pp.shape[1]))
    E_u = X_opt_pp[:,1:,1:] # unital submatrices
    unitarities = np.real(np.einsum('ijk, ijk -> i', E_u.conj(), E_u)/(pdim**2-1))
    return unitarities

# Spectrum of the Choi matrix
def generate_Choi_EV_table(X_opt, n_evals, gate_labels, filename):
    d,r,_ = X_opt.shape
    pdim = int(np.sqrt(r))
    Choi_evals_result = np.zeros((d,r))
    X_choi = X_opt.reshape(d,pdim,pdim,pdim,pdim)
    X_choi = np.einsum('ijklm->iljmk',X_choi).reshape(d,pdim**2,pdim**2)
    for j in range(d):
        Choi_evals_result[j,:] =  np.sort(np.abs(la.eig(X_choi[j])[0]))[::-1]
    Choi_evals_normalized = np.einsum('ij,i -> ij', Choi_evals_result,1/la.norm(Choi_evals_result, axis = 1, ord = 1))

    Choi_evals_normalized_str = [[number_to_str(Choi_evals_normalized[i,j], precision = 5) for i in range(d)] for j in range(n_evals)]
    df_g_evals = pd.DataFrame(Choi_evals_normalized_str).T
    df_g_evals.rename(index=gate_labels, inplace = True)

    df_g_evals.style.to_latex(filename+'.tex', column_format = 'c|*{%i}{c}'%n_evals, position_float = 'centering', 
                           hrules = True, caption = 'Eigenvalues of the Choi state', position = 'h!')
    
    
def local_dephasing_pp(prob_vec):
    D_loc_array = np.array([[[1,0,0,0],[0,1-2*p,0,0],[0,0,1-2*p,0],[0,0,0,1]] for p in prob_vec])
    return multikron(D_loc_array)

def dephasing_dist(prob_vec, X):
    X_deph = local_dephasing_pp(prob_vec)
    return la.norm(X-X_deph)

def dephasing_probabilities_2q(X_opt_pp, X_ideal_pp):
    dephasing_probs = []
    for i in range(X_opt_pp.shape[0]):
        dephasing_probs.append(minimize(dephasing_dist, [0.1,0.1], args = (X_opt_pp[i]@la.inv(X_ideal_pp[i]))).x)
    return dephasing_probs
        

from argparse import Namespace
def bootstrap_errors(K, X, E, rho, mGST_args, bootstrap_samples, weights, gate_labels, target_mdl, parametric = False):
    ns = Namespace(**mGST_args)
    if parametric:
        y = np.real(np.array([[E[i].conj()@low_level_jit.contract(X,j)@rho for j in J] for i in range(n_povm)]))
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
        df_g, df_o, _, _ = report(X_opt, E_opt, rho_opt, ns.J, y_sampled, target_mdl, gate_labels)
        df_g_list.append(df_g.values)
        df_o_list.append(df_o.values)

        X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)
        
        X_array[i, :] = X_opt_pp
        E_array[i, :] = E_opt_pp
        rho_array[i, :] = rho_opt_pp
        
    return (X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list))

def outcome_probs_from_files(folder_name, basis_dict, n_povm,N):
    
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
    return y/total_counts, np.max(total_counts)

def save_var_latex(key, value):
    # Saves variables in data file for to be read in latex document
    # Credit to https://stackoverflow.com/a/66620671
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
    # Order: gates + stat + povm - povm_freedom - gauge freedom - Kraus freedom
    return d*(pdim**2*(2*rK - 1) - rK**2) + pdim*(n_povm*pdim - pdim)