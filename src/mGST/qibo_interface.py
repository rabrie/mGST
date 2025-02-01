import numpy as np

from mGST import additional_fns, low_level_jit, algorithm
from mGST.compatibility import arrays_to_pygsti_model
from mGST.reporting.reporting import gauge_opt, quick_report

from qibo import Circuit, gates
from qibo.backends import _check_backend


def qibo_gate_to_operator(gate_set, backend=None):
    """Convert a set of Qibo gates to their unitary operators.

    This function takes a list of Qibo gate objects and them into operators, 
    returned as a NumPy array of matrices.

    Parameters
    ----------
    gate_set : list
        A list of Qibo gate objects. Each gate in the list is converted to its
        corresponding unitary.

    Returns
    -------
    numpy.ndarray
        An array of process matrices. Each element in the array is a 2D NumPy array.
    """
    backend = _check_backend(backend)

    return backend.cast([[gate.matrix(backend)] for gate in gate_set])

def add_idle_gates(gate_set, active_qubits, gate_qubits):
    """ Add additional idle gates to a gate set
    Each gate in the output gate_set acts on exactly the number of qubits
    on which the GST experiment ist defined. For instance if the GST
    experiment is supposed to be run on qubits [0,1,2], then a X-Gate
    on the 0-th qubit turns into X otimes Idle otimes Idle.
    This representation is needed for the internal handling in mGST.

    Parameters
    ----------
    gate_set : list of Qibo Circuits
        Each element is a gate to be used for GST, stored as a circuit.
    active_qubits : list integers
        Specifies the list of qubits on which GST-circuits are run.
    gate_qubits : list of lists of integers
        The i-th elements in the active_qubit list specifies on which qubits
        the original i-th gate in the input gate set is supposed to act.
    Returns
    -------
    gate_set : list of Qibo Circuits
        The output gate set with added idle gates.
    """
    for i in range(len(gate_qubits)):
        idle_qubits = active_qubits.copy()
        for j in gate_qubits[i]:
            idle_qubits.remove(j)
        if idle_qubits:
            for l in idle_qubits:
                gate_set[i].append(gates.I(l))

    return gate_set

def remove_idle_wires(qc):
    """ Removes all wires on which no gate acts
    Credit to: https://quantumcomputing.stackexchange.com/a/37192
    This shrinks the circuit for the conversion to quantum channels.

    Parameters
    ----------
    qc  Qibo quantum circuit

    Returns
    -------
    qc_out Qibo quantum circuit
        The circuit with idle wires removed.
    """
    qc_out = qc.copy()
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)

    return qc_out


def get_qibo_circuits(gate_sequences, gate_set, nqubits, active_qubits, **kwargs):
    """
    Generate a set of Qibo quantum circuits from specified gate sequences.

    This function creates a list of quantum circuits based on the provided sequences
    of gate indices. Each gate index corresponds to a gate in the provided `gate_set`.
    The gates are appended to a quantum circuit of a specified length and then measured.

    Parameters
    ----------
    gate_sequences : list of list of int
        A list where each element is a sequence of integers representing gate indices. Each integer
        corresponds to a gate in `gate_set`.
    gate_set : list
        A list of Qibo gate objects. The indices in `gate_sequences` refer to gates in this list.
    n_qubits : int
        The total number of qubits in the system.
    active_qubits : list of int
        The qubits on which the circuits are run.


    Returns
    -------
    list of QuantumCircuit
        A list of Qibo Circuit objects. Each circuit corresponds to one sequence in
        `gate_sequences`, with gates applied to the first qubit.
    """
    if isinstance(active_qubits, int):
        active_qubits = (active_qubits,)

    circuits = []
    for gate_sequence in gate_sequences:
        qc = Circuit(nqubits, **kwargs)
        for gate_num in gate_sequence:
            gate = gate_set[gate_num]
            gate = gate.on_qubits(dict(zip(gate.qubits, active_qubits)))
            qc.add(gate)
        qc.add(gates.M(qubit) for qubit in active_qubits)
        circuits.append(qc)

    return circuits


def get_gate_sequence(sequence_number, sequence_length, gate_set_length, seed=None):
    """Generate a set of random gate sequences.

    This function creates a specified number of random gate sequences, each of a given length.
    The gates are represented by numerical indices corresponding to elements in `gate_set`.
    Each sequence is a random combination of these indices.

    Parameters
    ----------
    sequence_number : int
        The number of gate sequences to generate.
    sequence_length : int
        The length of each gate sequence.
    gate_set_length : int
        The length of the set of gates to be used

    Returns
    -------
    numpy.ndarray
        An array of shape (sequence_number, sequence_length), where each row represents a
        randomly generated gate sequence.
    """
    rng = np.random.default_rng(seed)
    J_rand = rng.integers(0, int(gate_set_length**sequence_length) - 1, sequence_number)
    gate_sequences = np.array(
        [
            low_level_jit.local_basis(ind, gate_set_length, sequence_length)
            for ind in J_rand
        ]
    )

    return gate_sequences

def job_counts_to_mgst_format(active_qubits, n_povm, result_dict):
    """ Turns the dictionary of outcomes obtained from Qibo backend
        into the format which is used in mGST

    Parameters
    ----------
    active_qubits : list of int
        The qubits on which the circuits are run.
    n_povm : int
        Number of measurement outcomes, n_povm = physical dimension for basis measurements
    result_dict: (dict of str: int)
        Dictionary of outcomes from circuits run in a job
    Returns
    -------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence

    """
    basis_dict_list = []
    for result in result_dict:
        # Translate dictionary entries of bitstring on the full system to the decimal representation of bitstrings on the active qubits
        basis_dict = {entry: int("".join([entry[::-1][i] for i in active_qubits][::-1]), 2) for entry in result}
        # Sort by index:
        basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))
        basis_dict_list.append(basis_dict)
    y = []
    for i in range(len(result_dict)):
        row = [result_dict[i][key] for key in basis_dict_list[i]]
        if len(row) < n_povm:
            missing_entries = list(np.arange(n_povm))
            for given_entry in basis_dict_list[i].values():
                missing_entries.remove(given_entry)
            for missing_entry in missing_entries:
                row.insert(missing_entry, 0)  # 0 measurement outcomes in not recorded entry
        y.append(row / np.sum(row))
    y = np.array(y).T
    return y


def get_gate_estimation(gate_set, gate_sequences, sequence_results, shots, rK = 4):
    """Estimate quantum gates using a modified Gate Set Tomography (mGST) algorithm.

    This function simulates quantum gates, applies noise, and then uses the mGST algorithm
    to estimate the gates. It calculates and prints the Mean Variation Error (MVE) of the
    estimation.

    Parameters
    ----------
    gate_set : array_like
        The set of quantum gates to be estimated.
    gate_sequences : array_like
        The sequences of gates applied in the quantum circuit.
    sequence_results : array_like
        The results of executing the gate sequences.
    shots : int
        The number of shots (repetitions) for each measurement.
    """

    K_target = qibo_gate_to_operator(gate_set)
    gate_set_length = len(gate_set)
    pdim = K_target.shape[-1] # Physical dimension
    r = pdim ** 2   # Matrix dimension of gate superoperators
    n_povm = pdim   # Number of POVM elements
    sequence_length = gate_sequences.shape[0]

    # tensor of superoperators
    X_target = np.einsum("ijkl,ijnm -> iknlm", K_target, K_target.conj())
    X_target = np.reshape(X_target, (gate_set_length, pdim ** 2, pdim ** 2))

    # Initial state |0>
    rho_target = np.kron(
        additional_fns.basis(pdim, 0).T.conj(), 
        additional_fns.basis(pdim, 0),
    )
    rho_target = rho_target.reshape(-1).astype(complex)

    # Computational basis measurement:
    E_target = np.array(
        [
            np.kron(
                additional_fns.basis(pdim, i).T.conj(), 
                additional_fns.basis(pdim, i),
                ).reshape(-1)
            for i in range(pdim)
        ]
    ).astype(complex)
    target_mdl = arrays_to_pygsti_model(X_target, E_target, rho_target, basis="std")

    K_init = additional_fns.perturbed_target_init(X_target, rK)

    bsize = 30*pdim  # Batch size for optimization
    K, X, E, rho, res_list = algorithm.run_mGST(
        sequence_results, 
        gate_sequences,
        sequence_length, 
        gate_set_length, 
        r,
        rK, 
        n_povm, 
        bsize, 
        shots,
        method='SFN', 
        max_inits=10,
        max_iter=100, 
        final_iter=50,
        target_rel_prec=1e-4, 
        init=[K_init, E_target, rho_target]
    )

    # Output the final mean variation error
    mean_var_error = additional_fns.MVE(
        X_target, 
        E_target, 
        rho_target, 
        X, 
        E, 
        rho,
        gate_set_length, 
        sequence_length, 
        n_povm
    )
    mean_var_error = mean_var_error[0]
    print('Mean variation error:', mean_var_error)
    print('Optimizing gauge...')
    weights = dict({'G%i' % i: 1 for i in range(gate_set_length)}, **{'spam': 1})
    X_opt, E_opt, rho_opt = gauge_opt(X, E, rho, target_mdl, weights)
    print('Compressive GST routine complete')

    # Making sense of the outcomes
    df_g, df_o = quick_report(
        X_opt, 
        E_opt, 
        rho_opt, 
        gate_sequences, 
        sequence_results, 
        target_mdl
    )
    print('First results:')
    print(df_g.to_string())
    print(df_o.T.to_string())

    return X_opt, E_opt, rho_opt
