import numpy as np
import pygsti
from pygsti.report import reportables as rptbl
from pygsti.baseobjs import Label
from pygsti.tools import change_basis


def pygsti_model_to_arrays(model,basis = 'pp'):  
    """!

    Turns the gate set of a pygsti model into numpy arrays used by mGST

    Parameters
    -------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism

    basis : {'pp','std'} 
        The basis in which the output gate set will be given. It can be either
        'pp' (Pauli basis) or 'std' (standard basis)

    Returns
    -------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2)
    rho : numpy array
        Initial state vector of shape (dimension^2)

    """
    X = []
    op_Labels = [label for label in model.__dict__['operations'].keys()]
    effect_Labels = [label for label in model['Mdefault'].keys()]
    E = np.array([model['Mdefault'][label].to_dense().reshape(-1) for label in effect_Labels])
    rho = model['rho0'].to_dense().reshape(-1)
    for op_Label in op_Labels:
        X.append(model[op_Label].to_dense())
    if basis == 'pp':
        return np.array(X).astype(np.complex128), E.astype(np.complex128), rho.astype(np.complex128)
    if basis == 'std':
        return pp2std(np.array(X),E,rho)

def average_gate_fidelities(model1,model2,pdim, basis_string = 'pp'):
    """!

    Returns the average gate fidelities between gates of two pygsti models

    Parameters
    -------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    model2 : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    pdim : int
        physical dimension
    basis : {'pp','std'} 
        The basis in which the input models are gíven. It can be either
        'pp' (Pauli basis) or 'std' (standard basis, Default)

    Returns
    -------
    fidelities : 1D numpy array
        Array containing the average gate fidelities for all gates
        
    """
    ent_fids = []
    basis = pygsti.obj.Basis.cast(basis_string,pdim**2)
    labels1 = [label for label in model1.__dict__['operations'].keys()]
    labels2 = [label for label in model2.__dict__['operations'].keys()]

    for i in range(len(labels1)):
        ent_fids.append(float(rptbl.entanglement_fidelity(model1[labels1[i]], model2[labels2[i]], basis)))
    fidelities = (np.array(ent_fids)*pdim+1)/(pdim+1)
    return fidelities

def model_agfs(model,pdim):
    """!

    Returns the average gate fidelities between gates of two pygsti models

    Parameters
    -------
    model : pygsti ExplicitOpModel object
        Ĉontains the model parameters in the Pauli transfer matrix formalism
    pdim : int
        physical dimension

    Returns
    -------
    fidelities : 1D numpy array
        Average gate fidelities between all different gates of the input model
        
    """
    ent_fids = [] 
    basis = pygsti.obj.Basis.cast("pp",pdim**2)
    labels = [label for label in model.__dict__['operations'].keys()]    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if j>i:
                ent_fids.append(float(rptbl.entanglement_fidelity(model[labels[i]], model[labels[j]], basis)))
    fidelities = (np.array(ent_fids)*pdim+1)/(pdim+1)
    return fidelities

def arrays_to_pygsti_model(X,E,rho, basis = 'std'): #pygsti model is by default in Pauli-basis
    """!

    Turns a gate set given by numpy arrays into a pygsti model

    Parameters
    -------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2)
    rho : numpy array
        Initial state vector of shape (dimension^2)
    basis : {'pp','std'} 
        The basis in which the INPUT gate set is given. It can be either
        'pp' (Pauli basis) or 'std' (standard basis)

    Returns
    -------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism

    """
    d = X.shape[0]
    pdim = int(np.sqrt(len(rho)))
    Id = np.zeros(pdim**2)
    Id[0] = 1
    effect_label_str = ['%i'%k for k in range(E.shape[0])]
    if basis == 'std':
        X,E,rho = std2pp(X,E,rho)
    mdl_out = pygsti.construction.build_explicit_model( 
        [i for i in range(int(np.log(pdim)/np.log(2)))],[Label('G%i'%i) for i in range(d)], 
        [':'.join(['I(%i)'%i for i in range(int(np.log(pdim)/np.log(2)))]) for l in range(d)],
        effectLabels=effect_label_str)
    mdl_out['rho0'] = np.real(rho)
    for i in range(E.shape[0]):
        mdl_out['Mdefault']['%i'%i] = np.real(E[i])
    for i in range(d):
        mdl_out[Label('G%i'%i)] = np.real(X[i].T)
    return mdl_out

def std2pp(X,E,rho):
    """!

    Basis change of an mGST model from the standard basis to the Pauli basis

    Parameters
    -------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in standard basis
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in standard basis
    rho : numpy array
        Initial state vector of shape (dimension^2) in standard basis

    Returns
    -------
    Xpp : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in Pauli basis
    Epp : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
    rhopp : numpy array
        Initial state vector of shape (dimension^2) in Pauli basis

    """
    Xpp = np.array([np.array(change_basis(X[i],'std','pp')) for i in range(X.shape[0])])
    Epp = np.array([np.array(change_basis(E[i],'std','pp')) for i in range(E.shape[0])])
    return Xpp.astype(np.complex128),Epp.astype(np.complex128),change_basis(rho,'std','pp').astype(np.complex128)

def pp2std(X,E,rho):
    """!

    Basis change of an mGST model from the Pauli basis to the standard basis

    Parameters
    -------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in Pauli basis
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
    rho : numpy array
        Initial state vector of shape (dimension^2) in Pauli basis

    Returns
    -------
    Xstd : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2) in standard basis
    Estd : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in standard basis
    rhostd : numpy array
        Initial state vector of shape (dimension^2) in standard

    """
    Xstd = np.array([np.array(change_basis(X[i],'pp','std')) for i in range(X.shape[0])])
    Estd = np.array([np.array(change_basis(E[i],'pp','std')) for i in range(E.shape[0])])
    return Xstd.astype(np.complex128),Estd.astype(np.complex128),change_basis(rho,'pp','std').astype(np.complex128)

def pygstiExp_to_list(model,max_germ_len):
    """!

    Takes the meausurement sequences of a pygsti model and turns them into a sequence list for mGST

    Parameters
    -------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    max_germ_len : int
        Maximum number of germ repetitions in the pygsti circuit design

    Returns
    -------
    J_GST : numpy array 
        2D array where each row contains the gate indices of a gate sequence
        
    """
    prep_fiducials = model.prep_fiducials() 
    meas_fiducials = model.meas_fiducials() 
    germs = model.germs()                    
    maxLengths = [max_germ_len]
    exp_design = pygsti.protocols.StandardGSTDesign(model.target_model(), prep_fiducials, meas_fiducials,germs, maxLengths)
    listOfExperiments = pygsti.circuits.create_lsgst_circuits(
        model.target_model(), prep_fiducials, meas_fiducials, germs, maxLengths)
    op_Labels = [label for label in model.target_model().__dict__['operations'].keys()]
    exp_list = []
    max_length = max([len(x.to_pythonstr(op_Labels)) for x in listOfExperiments])
    for i in range(len(listOfExperiments)):
        exp = listOfExperiments[i].to_pythonstr(op_Labels)
        gate_numbers = [ord(x)-97 for x in exp.lower()]
        gate_numbers = np.pad(gate_numbers,(0,max_length-len(gate_numbers)),'constant',constant_values=-1)
        exp_list.append(gate_numbers)

    J_GST = np.array([[int(exp_list[i][j]) for j in range(max_length)] for i in range(len(exp_list))])
    return J_GST

