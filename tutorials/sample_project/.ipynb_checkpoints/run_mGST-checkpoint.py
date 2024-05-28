from mGST import compatibility,algorithm, optimization, low_level_jit, additional_fns
from mGST.reporting.plot_params import *
from mGST.reporting import uncertainty, reporting
import pickle, sys
from argparse import Namespace
from pandas import read_csv
from os import listdir
from itertools import product
from pandas import DataFrame
from pygsti.tools import change_basis
from subprocess import call
from exp_design.exp_description import *


# Rank of the gate superoperators
r = pdim**2   

# Hyperparameter of the optimization, which determines how many sequences are used at each iteration.
# This can be left as is in most cases.
bsize = 30*pdim 

filename = 'mGST_results/results_'+experiment_name


# ###################################### Import sequence list and load data
J_list = read_csv("exp_design/sequences.csv", delimiter=",", names = list(range(32))).values
N = J_list.shape[0] + 1
J_list = [[int(x) for x in J_list[i,:] if str(x) != 'nan'] for i in range(N-1)]
J_list.insert(0,[])
l_max = np.max([len(J_list[i]) for i in range(N)])

J = []
for i in range(N):
    J.append(list(np.pad(J_list[i],(0,l_max-len(J_list[i])),'constant',constant_values=-1)))
J = np.array(J).astype(int)[:,::-1]
y, meas_samples = reporting.outcome_probs_from_files(folder_name, basis_dict, n_povm,N)

mGST_args = {'y': y, 'J': J, 'l': l_max, 'd': d, 'r': r, 'rK': rK, 'n_povm': n_povm, 'bsize': bsize, 
             'meas_samples': meas_samples, 'method': 'SFN', 'max_inits': 10, 'max_iter': 100, 
             'final_iter': 100, 'threshold_multiplier': 5, 'target_rel_prec': 5e-6}
ns = Namespace(**mGST_args)

K_target = additional_fns.Kraus_rep(X_target,d,pdim,rK)

# ###################################### mGST

if from_init:
    K,X,E,rho,res_list = algorithm.run_mGST(ns.y,ns.J,ns.l,ns.d,ns.r,ns.rK, ns.n_povm, ns.bsize, ns.meas_samples, method = ns.method,
                     max_inits = ns.max_inits, max_iter = 100, final_iter = ns.final_iter, threshold_multiplier = ns.threshold_multiplier, 
                     target_rel_prec = ns.target_rel_prec, init = [K_target, E_target, rho_target], testing = False)   
else:
    K,X,E,rho,res_list = algorithm.run_mGST(ns.y,ns.J,ns.l,ns.d,ns.r,ns.rK, ns.n_povm, ns.bsize, ns.meas_samples, method = ns.method,
                     max_inits = ns.max_inits, max_iter = ns.max_iter, final_iter = ns.final_iter, threshold_multiplier = ns.threshold_multiplier, 
                     target_rel_prec = ns.target_rel_prec, testing = False)   

# ###################################### Show preliminary results in console 
    
target_mdl = compatibility.arrays_to_pygsti_model(X_target,E_target,rho_target, basis = 'std')

weights = {'G0': 1,'G1': 1, 'G2': 1, 'G3': 1, 'G4': 1, 'G5': 1, 'spam': 1}
X_opt, E_opt, rho_opt = reporting.gauge_opt(X, E, rho, target_mdl, weights)
df_g, df_o, s_g, s_o = reporting.quick_report(X_opt, E_opt, rho_opt, J, y, target_mdl, gate_labels)
print('#################')
print('First results:')
print(df_g.to_string())
print(df_o.T.to_string())  

# ###################################### Save data
print('\n')
bootstrap_gen = input('Should bootstrapping error bars be generated (this can take a while)? (y/n)')

if bootstrap_gen in ['Y', 'y', 'Yes', 'yes', 'YES']:
    bootstrap_samples = int(input('Please set the number of bootstrapping runs (recommended = 50):'))
    X_array, E_array, rho_array, df_g_array, df_o_array = reporting.bootstrap_errors(K ,X ,E ,rho, mGST_args, bootstrap_samples, weights, gate_labels, target_mdl, parametric = False)
    results = {'estimates': (K,X,E,rho), 'gauge_opt_estimates': (X_opt, E_opt, rho_opt), 'bootstrap_data': (X_array, E_array, rho_array, df_g_array, df_o_array)}
else:
    results = {'estimates': (K,X,E,rho), 'gauge_opt_estimates': (X_opt, E_opt, rho_opt), 'quick_report': (df_g, df_o)}
    
data_to_safe = {'parameters': mGST_args, 'results': results}
with open(filename, 'wb') as handle:
    pickle.dump(data_to_safe, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Save experiment description for latex document
reporting.save_var_latex("experiment_name", experiment_name)
reporting.save_var_latex("experiment_date", experiment_date)
reporting.save_var_latex("rK", rK)
reporting.save_var_latex('free_params', int(reporting.n_params(pdim, ns.d, ns.rK, ns.n_povm)))
reporting.save_var_latex("meas_samples", int(meas_samples))
reporting.save_var_latex("gate_labels", gate_labels)

# ###################################### Report generation prompt
print('\n')
report_gen = input('Would you like to generate gate plots and extended error tables? (y/n)')

if report_gen in ['Y', 'y', 'Yes', 'yes', 'YES']:
    print('Results are being generated and saved in the /mGST_results/ folder...')
else:
    sys.exit()

    
# ###################################### Load data
    
with open(filename, 'rb') as handle:
    b = pickle.load(handle)
results = b['results']


# ###################################### Generating gate and SPAM error table

Pauli_labels_loc = ['I','X','Y','Z']
Pauli_labels_rep = [Pauli_labels_loc for _ in range(int(np.log2(pdim)))]
separator = ''
Pauli_labels = [separator.join(map(str,x)) for x in product(*Pauli_labels_rep)]

std_labels_loc = ['0','1']
std_labels_rep = [std_labels_loc for _ in range(int(np.log2(pdim)))]
separator = ''
std_labels = [separator.join(map(str,x)) for x in product(*std_labels_rep)]

X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)
X_target_pp, E_target_pp, rho_target_pp = compatibility.std2pp(X_target, E_target, rho_target)

df_g, df_o, s_g, s_o = reporting.report(X_opt, E_opt, rho_opt, J, y, target_mdl, gate_labels)



if bootstrap_gen in ['Y', 'y', 'Yes', 'yes', 'YES']:
    X_array, E_array, rho_array, df_g_array, df_o_array = b['results']['bootstrap_data']
    df_g_array[df_g_array == -1] = np.nan
    df_o_array[df_o_array == -1] = np.nan
    
    gate_errors = np.array([df_g.values for _ in range(bootstrap_samples)]) - df_g_array
    other_errors = np.array([df_o.values for _ in range(bootstrap_samples)]) - df_o_array


    percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5,97.5] , axis = 0)
    percentiles_o_low, percentiles_o_high = np.nanpercentile(df_o_array, [2.5,97.5] , axis = 0)

    bootstrap_unitarities = np.array([reporting.unitarities(X_array[i]) for i in range(bootstrap_samples)])
    percentiles_u_low, percentiles_u_high = np.nanpercentile(bootstrap_unitarities, [2.5,97.5] , axis = 0)



    if rK == 1:
        df_g_err = DataFrame({
            r"\shortstack{Average gate fidelity \\ $\mathcal{F}_{\mathrm{avg}}(\mathcal{U}_i, \hat{\mathcal{G}}_i)$}": 
                [uncertainty.number_to_str(df_g.values[i,0], [percentiles_g_high[i,0], percentiles_g_low[i,0]], precision = 4) for i in range(len(gate_labels))],
            r"\shortstack{Diamond distance \\ $\frac{1}{2}||\mathcal{U}_i - \hat{\mathcal{G}}_i||_{\diamond}$}": 
                [uncertainty.number_to_str(df_g.values[i,1], [percentiles_g_high[i,1], percentiles_g_low[i,1]], precision = 4) for i in range(len(gate_labels))],
        })
        U_opt = reporting.phase_opt(X_opt,K_target)

        angles, axes, pauli_coeffs  = reporting.compute_angles_axes(U_opt)
        angles_t, axes_t, pauli_coeffs_t = reporting.compute_angles_axes(K_target[:,0,:,:])

        ax_ang_estimate = np.concatenate((np.array(angles).reshape(d,1), axes), axis = 1)
        estimate_ax_array = np.array([axes for _ in range(bootstrap_samples)])


        bootstrap_ax_ang = np.zeros((len(X_array),d,r))
        bootstrap_pauli_coeffs = np.zeros((len(X_array),d,r-1))
        for i in range(len(X_array)):
            U_opt_ = reporting.phase_opt(np.array([change_basis(X_array[i][j],'pp','std') for j in range(d)]),K_target)
            angles_, axes_, pp_vec = reporting.compute_angles_axes(U_opt_)   
            bootstrap_ax_ang[i,:,0] = angles_
            bootstrap_ax_ang[i,:,1:] = axes_
            bootstrap_pauli_coeffs[i,:,:] = pp_vec
        ax_ang_percentiles_low, ax_ang_percentiles_high = np.nanpercentile(bootstrap_ax_ang,  [2.5,97.5], axis = 0)
        pauli_coeffs_low, pauli_coeffs_high = np.nanpercentile(bootstrap_pauli_coeffs,  [2.5,97.5], axis = 0)

        ## Table of axes parameters
        df_g_rotation = DataFrame(np.array([[uncertainty.number_to_str(ax_ang_estimate[i,j], 
                                                     [ax_ang_percentiles_high[i,j], ax_ang_percentiles_low[i,j]], precision = 3) 
                                                for i in range(len(gate_labels))] for j in range(r)]).T)

        col_labels = [r'$n_{%s}$'%label for label in Pauli_labels]
        column_labels = col_labels #Pauli_labels.copy()
        column_labels[0] = r"$\alpha/\pi$"
        df_g_rotation.columns = column_labels

        df_g_rotation.rename(index=gate_labels, inplace = True)

        df_g_rotation.T.style.to_latex('mGST_results/bloch_rotation_axes_coeffs.tex', column_format = 'c|*{%i}{c}'%(d), position_float = 'centering', 
                                hrules = True, caption = 'Normalized rotation axes coefficient. Errors \
                                correspond to the 95th percentile over %i bootstrapping runs.'%bootstrap_samples, position = 'h!')
    else:
        df_g_err = DataFrame({
            r"\shortstack{Average gate fidelity \\ $\mathcal{F}_{\mathrm{avg}}(\mathcal{U}_i, \hat{\mathcal{G}}_i)$}": 
                [uncertainty.number_to_str(df_g.values[i,0], [percentiles_g_high[i,0], percentiles_g_low[i,0]], precision = 4) for i in range(len(gate_labels))],
            r"\shortstack{Diamond distance \\ $\frac{1}{2}||\mathcal{U}_i - \hat{\mathcal{G}}_i||_{\diamond}$}": 
                [uncertainty.number_to_str(df_g.values[i,1], [percentiles_g_high[i,1], percentiles_g_low[i,1]], precision = 4) for i in range(len(gate_labels))],
            r"\shortstack{Unitarity \\ $u(\hat{\mathcal G}_i)$}": 
                [uncertainty.number_to_str(reporting.unitarities(X_opt_pp)[i], [percentiles_u_high[i], percentiles_u_low[i]],  precision = 4) for i in range(len(gate_labels))],
        })        


    df_o_err = DataFrame({
        "Final cost": uncertainty.number_to_str(df_o.values[0,0], [percentiles_o_high[0,0], percentiles_o_low[0,0]], precision = 4),
        "Mean TVD: estimate - data": uncertainty.number_to_str(df_o.values[0,1], [percentiles_o_high[0,1], percentiles_o_low[0,1]], precision = 4),
        "Mean TVD: target - data": uncertainty.number_to_str(df_o.values[0,2], [percentiles_o_high[0,2], percentiles_o_low[0,2]], precision = 4),
        "POVM - diamond dist.": uncertainty.number_to_str(df_o.values[0,3], [percentiles_o_high[0,3], percentiles_o_low[0,3]], precision = 4),
        "State - trace dist.": uncertainty.number_to_str(df_o.values[0,4], [percentiles_o_high[0,4], percentiles_o_low[0,4]], precision = 4),  
    }, index = [0])
    df_g_err.rename(index=gate_labels, inplace = True)
    df_o_err.rename(index={0: ""}, inplace = True)



    df_g_err.style.to_latex('mGST_results/gate_errors.tex', column_format = 'c|c|c|c|c', position_float = 'centering', 
                            hrules = True, caption = 'Gate quality measures with errors corresponding to the \
                            95th percentile over %i bootstrapping runs.'%bootstrap_samples, position = 'h!')
    df_o_err.style.to_latex('mGST_results/spam_errors.tex', column_format = 'c|c|c|c|c|c', position_float = 'centering', 
                            hrules = True, caption = 'State and measurement quality measures with errors corresponding to the \
                            95th percentile over %i bootstrapping runs.'%bootstrap_samples, position = 'h!')
    

else:
    if rK ==1:
        df_g_err = DataFrame({
            "Average gate Fidelity": [uncertainty.number_to_str(df_g.values[i,0], precision = 4) for i in range(len(gate_labels))],
            "Diamond distance": [uncertainty.number_to_str(df_g.values[i,1],  precision = 4) for i in range(len(gate_labels))],
        })
        U_opt = reporting.phase_opt(X_opt,K_target)

        angles, axes, pauli_coeffs  = reporting.compute_angles_axes(U_opt)
        angles_t, axes_t, pauli_coeffs_t = reporting.compute_angles_axes(K_target[:,0,:,:])

        ax_ang_estimate = np.concatenate((np.array(angles).reshape(d,1), axes), axis = 1)

        ## Table of axes parameters
        df_g_rotation = DataFrame(np.array(
            [[uncertainty.number_to_str(ax_ang_estimate[i,j], precision = 3) for i in range(len(gate_labels))] for j in range(r)]).T)

        col_labels = [r'$n_{%s}$'%label for label in Pauli_labels]
        column_labels = col_labels #Pauli_labels.copy()
        column_labels[0] = r"$\alpha/\pi$"
        df_g_rotation.columns = column_labels

        df_g_rotation.rename(index=gate_labels, inplace = True)

        df_g_rotation.T.style.to_latex('mGST_results/bloch_rotation_axes_coeffs.tex', column_format = 'c|*{%i}{c}'%(d), position_float = 'centering', 
                                hrules = True, caption = 'Normalized rotation axes coefficient.', position = 'h!')
    else:
            df_g_err = DataFrame({
        "Average gate Fidelity": [uncertainty.number_to_str(df_g.values[i,0], precision = 4) for i in range(len(gate_labels))],
        "Diamond distance": [uncertainty.number_to_str(df_g.values[i,1],  precision = 4) for i in range(len(gate_labels))],
        "Unitarity": [uncertainty.number_to_str(reporting.unitarities(X_opt_pp)[i],  precision = 4) for i in range(len(gate_labels))],
        # "Entanglemen fidelity to depol. channel": [uncertainty.number_to_str(reporting.eff_depol_params(X_opt_pp)[i],  precision = 4) 
        #                                            for i in range(len(gate_labels))],
        # "Min. spectral distances": [number_to_str(df_g.values[i,2], precision = 4) for i in range(len(gate_labels))]
        })
    df_o_err = DataFrame({
        "Final cost": uncertainty.number_to_str(df_o.values[0,0], precision = 4),
        "Mean TVD: estimate - data": uncertainty.number_to_str(df_o.values[0,1],  precision = 4),
        "Mean TVD: target - data": uncertainty.number_to_str(df_o.values[0,2],  precision = 4),
        "POVM - diamond dist.": uncertainty.number_to_str(df_o.values[0,3],  precision = 4),
        "State - trace dist.": uncertainty.number_to_str(df_o.values[0,4],  precision = 4),  
    }, index = [0])
    df_g_err.rename(index=gate_labels, inplace = True)
    df_o_err.rename(index={0: ""}, inplace = True)


    # ## Saving gate and SPAM error tables

    df_g_err.style.to_latex('mGST_results/gate_errors.tex', column_format = 'c|c|c|c|c|c', position_float = 'centering', 
                            hrules = True, caption = 'Gate quality measures', position = 'h!')
    df_o_err.style.to_latex('mGST_results/spam_errors.tex', column_format = 'c|c|c|c|c|c', position_float = 'centering', 
                            hrules = True, caption = 'State and measurement quality measures', position = 'h!')



# ## Spectrum of the Choi matrix
reporting.generate_Choi_EV_table(X_opt, np.min([r,7]), gate_labels, 'mGST_results/Choi_evals')

# ###################################### Plots

# ## Gate plots in Pauli basis

gates = X_opt_pp
gates_target = X_target_pp
filename = 'mGST_results/pp'##
reporting.generate_gate_err_pdf(filename, gates, gates_target, basis_labels = Pauli_labels)



filename = 'mGST_results/spam_errs_std_real.pdf'
reporting.generate_spam_err_std_pdf(filename, E_opt.real, rho_opt.real,
                                E_target.real, rho_target.real,  basis_labels = std_labels)

filename = 'mGST_results/spam_errs_std_imag.pdf'
reporting.generate_spam_err_std_pdf(filename, E_opt.imag, rho_opt.imag,
                                E_target.imag, rho_target.imag,  basis_labels = std_labels)


print('\n')
tex_gen = input('Would you like to generate a latex report? (y/n)')

if tex_gen in ['Y', 'y', 'Yes', 'yes', 'YES']:
    print('Generating pdf in the /report/ folder...')
    print('Note: This requires latex to be installed.', 
    'If the output pdf contains errors, check the log file in /report/.')
else:
    sys.exit()
    
# ###################################### Run latex

call(['pdflatex', '--output-directory', 'report', '--interaction=batchmode', 'tex_report.tex'])
