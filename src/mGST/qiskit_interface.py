import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import IGate
from qiskit.quantum_info import Operator
import random
from scipy.linalg import expm
from mGST import additional_fns, low_level_jit, algorithm
import matplotlib.pyplot as plt


def qiskit_gate_to_operator(gate_set):
    """Convert a set of Qiskit gates to their unitary operators.

    This function takes a list of Qiskit gate objects and them into operators, returned as a NumPy
    array of matrices.

    Parameters
    ----------
    gate_set : list
        A list of Qiskit gate objects. Each gate in the list is converted to its
        corresponding unitary.

    Returns
    -------
    numpy.ndarray
        An array of process matrices. Each element in the array is a 2D NumPy array.
    """
    return np.array([[Operator(gate).to_matrix()] for gate in gate_set])

def add_idle_gates(gate_set, active_qubits, gate_qubits):
    """ Add additional idle gates to a gate set
    Each gate in the output gate_set acts on exactly the number of qubits
    on which the GST experiment ist defined. For instance if the GST
    experiment is supposed to be run on qubits [0,1,2], then a X-Gate
    on the 0-th qubit turns into X \otimes Idle \otimes Idle.
    This representation is needed for the internal handling in mGST.

    Parameters
    ----------
    gate_set : list of Qiskit Circuits
        Each element is a gate to be used for GST, stored as a circuit.
    active_qubits : list integers
        Specifies the list of qubits on which GST-circuits are run.
    gate_qubits : list of lists of integers
        The i-th elements in the active_qubit list specifies on which qubits
        the original i-th gate in the input gate set is supposed to act.
    Returns
    -------
    gate_set : list of Qiskit Circuits
        The output gate set with added idle gates.
    """
    d = len(gate_qubits)
    for i in range(d):
        idle_qubits = active_qubits.copy()
        for j in gate_qubits[i]:
            idle_qubits.remove(j)
        if idle_qubits:
            for l in idle_qubits:
                gate_set[i].append(IGate(), [l])
    return gate_set

def remove_idle_wires(qc):
    """ Removes all wires on which no gate acts
    Credit to: https://quantumcomputing.stackexchange.com/a/37192
    This shrinks the circuit for the conversion to quantum channels.

    Parameters
    ----------
    qc  Qiskit quantum circuit

    Returns
    -------
    qc_out Qiskit quantum circuit
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


def get_qiskit_circuits(gate_sequences, gate_set, n_qubits, active_qubits, measure_active=False):
    """
    Generate a set of Qiskit quantum circuits from specified gate sequences.

    This function creates a list of quantum circuits based on the provided sequences
    of gate indices. Each gate index corresponds to a gate in the provided `gate_set`.
    The gates are appended to a quantum circuit of a specified length and then measured.

    Parameters
    ----------
    gate_sequences : list of list of int
        A list where each element is a sequence of integers representing gate indices. Each integer
        corresponds to a gate in `gate_set`.
    gate_set : list
        A list of Qiskit gate objects. The indices in `gate_sequences` refer to gates in this list.
    n_qubits : int
        The total number of qubits in the system.
    active_qubits : list of int
        The qubits on which the circuits are run.
    measure_active : bool
        If set to "True", only the active qubits are measured, otherwise all qubits are measured.


    Returns
    -------
    list of QuantumCircuit
        A list of Qiskit QuantumCircuit objects. Each circuit corresponds to one sequence in
        `gate_sequences`, with gates applied to the first qubit.
    """
    qiskit_circuits = []

    for gate_sequence in gate_sequences:
        qc = QuantumCircuit(n_qubits,n_qubits)
        for gate_num in gate_sequence:
            qc.compose(gate_set[gate_num], active_qubits, inplace=True)
            qc.barrier(active_qubits)
        if measure_active:
            qc.measure(active_qubits, active_qubits)
        else:
            qc.measure_all()
        qiskit_circuits.append(qc)
    return qiskit_circuits


def simulate_circuit(qiskit_circuits, shots):
    """Simulate a list of quantum circuits using Qiskit's Aer qasm simulator

    This function takes a list of Qiskit QuantumCircuit objects and simulates each circuit using
    the specified number of shots. The function collects the simulation results and normalizes them
    by the total number of shots. The results are returned as a NumPy array.

    Parameters
    ----------
    qiskit_circuits : list
        A list of Qiskit QuantumCircuit objects to be simulated.
    shots : int
        The number of shots (repetitions) to use for each circuit simulation.

    Returns
    -------
    numpy.ndarray
        An array of normalized results. Each row in the array corresponds to a quantum circuit
        in `qiskit_circuits`, and each column corresponds to a possible measurement outcome.
        Values in the array are normalized by the total number of shots.
    """
    simulator = Aer.get_backend("qasm_simulator")
    sequence_results = simulator.run(qiskit_circuits, shots=shots).result().get_counts()

    results = [[], []]

    for item in sequence_results:
        try:
            results[0].append(item['0'] / shots)
        except KeyError:
            results[0].append(0.0)
        try:
            results[1].append(item['1'] / shots)
        except KeyError:
            results[1].append(0.0)
    return np.array(results)


def get_gate_sequence(sequence_number, sequence_length, gate_set_length):
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
    J_rand = np.array(random.sample(range(gate_set_length**sequence_length), sequence_number))
    gate_sequences = np.array([low_level_jit.local_basis(ind, gate_set_length, sequence_length)
                               for ind in J_rand])
    return gate_sequences


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

    K_target = qiskit_gate_to_operator(gate_set)
    gate_set_length = len(gate_set)
    pdim = K_target.shape[-1] # Physical dimension
    r = pdim ** 2   # Matrix dimension of gate superoperators
    n_povm = pdim   # Number of POVM elements
    sequence_length = gate_sequences.shape[0]

    X_target = np.einsum('ijkl,ijnm -> iknlm', K_target, K_target.conj()
                         ).reshape(gate_set_length, pdim ** 2, pdim ** 2)  # tensor of superoperators

    # Initial state |0>
    rho_target = np.kron(additional_fns.basis(pdim, 0).T.conj(), additional_fns.basis(pdim, 0)).reshape(-1).astype(np.complex128)

    # Computational basis measurement:
    E_target = np.array(
        [np.kron(additional_fns.basis(pdim, i).T.conj(), additional_fns.basis(pdim, i)).reshape(-1)
         for i in range(pdim)]
    ).astype(np.complex128)

    K_init = additional_fns.perturbed_target_init(X_target, rK)

    bsize = 30*pdim  # Batch size for optimization
    K, X, E, rho, res_list = algorithm.run_mGST(sequence_results, gate_sequences,
                                                sequence_length, gate_set_length, r,
                                                rK, n_povm, bsize, shots,
                                                method='SFN', max_inits=10,
                                                max_iter=100, final_iter=50,
                                                target_rel_prec=1e-4, init=[K_init, E_target, rho_target])

    # Output the final mean variation error
    mean_var_error = additional_fns.MVE(X_target, E_target, rho_target, X, E, rho,
                                        gate_set_length, sequence_length, n_povm)[0]
    print('Mean variation error:', mean_var_error)
    plt.semilogy(res_list)   # plot the objective function over the iterations
    plt.show()

    return K,X,E,rho