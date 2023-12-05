import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
import random
from scipy.linalg import expm
from mGST import additional_fns, low_level_jit, algorithm
import matplotlib.pyplot as plt


def qiskit_gate_to_kraus(gate_set):
    """Convert a set of Qiskit gates to their corresponding Kraus operators.

    This function takes a list of Qiskit gate objects and converts each gate into
    its Kraus operator representation. The Kraus operators are returned as a NumPy
    array of matrices.

    Parameters
    ----------
    gate_set : list
        A list of Qiskit gate objects. Each gate in the list is converted to its
        corresponding Kraus operator.

    Returns
    -------
    numpy.ndarray
        An array of Kraus operators. Each element in the array is a 2D NumPy array
        representing the Kraus operator of the corresponding gate in `gate_set`.
    """
    return np.array([[Operator(gate).data] for gate in gate_set])


def get_qiskit_circuits(gate_sequences, gate_set):
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

    Returns
    -------
    list of QuantumCircuit
        A list of Qiskit QuantumCircuit objects. Each circuit corresponds to one sequence in
        `gate_sequences`, with gates applied to the first qubit.
    """
    qiskit_circuits = []

    for gate_sequence in gate_sequences:
        qc = QuantumCircuit(1)
        for gate_num in gate_sequence:
            qc.append(gate_set[int(gate_num)], [0])
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
    sequence_results = execute(qiskit_circuits, simulator, shots=shots).result().get_counts()

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


def get_gate_estimation(gate_set, gate_sequences, gate_set_length, sequence_length,
                        sequence_results, shots):
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
    gate_set_length : int
        The number of gates in the gate set.
    sequence_length : int
        The length of each gate sequence.
    sequence_results : array_like
        The results of executing the gate sequences.
    shots : int
        The number of shots (repetitions) for each measurement.
    """

    pdim = 2   # Physical dimension
    r = pdim ** 2   # Matrix dimension of gate superoperators
    rK = 1   # Rank of the mGST model estimate
    n_povm = 2   # Number of POVM elements

    K_true = qiskit_gate_to_kraus(gate_set)
    X_true = np.einsum('ijkl,ijnm -> iknlm', K_true, K_true.conj()).reshape(
        gate_set_length, r, r)

    K_depol = additional_fns.depol(pdim, 0.02)
    G_depol = np.einsum('jkl,jnm -> knlm', K_depol, K_depol.conj()).reshape(r, r)
    rho_true = G_depol @ np.array([[1, 0], [0, 0]]).reshape(-1).astype(np.complex128)

    E1 = np.array([[1, 0], [0, 0]]).reshape(-1)
    E2 = np.array([[0, 0], [0, 1]]).reshape(-1)
    E_true = np.array([E1, E2]).astype(np.complex128)   # Full POVM

    delta = 0.1  # Unitary noise parameter

    # Generate noisy version of true gate set
    K0 = np.zeros((gate_set_length, rK, pdim, pdim)).astype(np.complex128)
    for i in range(gate_set_length):
        U_p = expm(delta * 1j * additional_fns.randHerm(pdim)).astype(np.complex128)
        K0[i] = np.einsum('jkl,lm', K_true[i], U_p)

    rho0 = additional_fns.randpsd(r).copy()
    A0 = additional_fns.randKrausSet(1, r, n_povm)[0].conj()
    E0 = np.array([(A0[i].T.conj() @ A0[i]).reshape(-1) for i in range(n_povm)]).copy()

    bsize = 50  # Batch size for optimization
    K, X, E, rho, res_list = algorithm.run_mGST(sequence_results, gate_sequences,
                                                sequence_length, gate_set_length, r,
                                                rK, n_povm, bsize, shots,
                                                method='SFN', max_inits=10,
                                                max_iter=30, final_iter=10,
                                                target_rel_prec=1e-4, init=[K0, E0, rho0])

    # Output the final mean variation error
    mean_var_error = additional_fns.MVE(X_true, E_true, rho_true, X, E, rho,
                                        gate_set_length, sequence_length, n_povm)[0]
    print('Mean variation error:', mean_var_error)
    plt.semilogy(res_list)   # plot the objective function over the iterations
    plt.show()
