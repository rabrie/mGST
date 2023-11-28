import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, execute
from qiskit.circuit.library import IGate
import random
from mGST import low_level_jit


def qiskit_gate_to_kraus(gate):
    """Convert a Qiskit gate to its equivalent Kraus operator.

    This function takes a Qiskit gate object and converts it into a Kraus operator
    represented by a numpy array. If the input gate is an identity gate (IGate),
    the function directly returns the 2x2 identity matrix. Otherwise, the function
    creates a Qiskit QuantumCircuit with the gate, transpiles the circuit for the
    unitary simulator, and then runs the simulator to extract the unitary matrix
    representing the Kraus operator.

    Parameters
    ----------
    gate : qiskit.circuit.Gate
        The Qiskit gate to be converted. This can be any gate that is supported by
        the Aer unitary simulator.

    Returns
    -------
    numpy.ndarray
        A 2x2 or larger unitary matrix (depending on the number of qubits for the gate)
        representing the Kraus operator equivalent of the input Qiskit gate.
    """
    # Directly return the identity matrix for the idle gate
    if isinstance(gate, IGate):
        return np.eye(2)

    # Create a quantum circuit with the gate
    qreg = QuantumRegister(gate.num_qubits)
    qc = QuantumCircuit(qreg)
    qc.append(gate, qreg)

    # Transpile for the unitary simulator using specific basis gates
    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "rzz"])

    # Use Aer's unitary simulator to find the unitary matrix
    simulator = Aer.get_backend("unitary_simulator")
    result = simulator.run(qc).result()
    unitary_matrix = result.get_unitary(qc)

    return unitary_matrix


def get_qiskit_circuits(sequence, gate_map):
    """Construct a QuantumCircuit in Qiskit from a given sequence of gate numbers.

    This function creates a QuantumCircuit using a sequence of gate numbers,
    where each number in the sequence corresponds to a gate in the provided gate map.
    Until now the function is intended for 1 qubit GST.

    Parameters
    ----------
    sequence : iterable
        An iterable (like a list or array) of gate numbers. Each number corresponds
        to a gate in `gate_map`. The gates will be applied in the order they appear
        in the sequence.
    gate_map : dict
        A dictionary mapping gate numbers (integers) to Qiskit gate objects. The keys
        are the numbers that appear in `sequence`, and the values are the Qiskit gates
        that those numbers correspond to.

    Returns
    -------
    QuantumCircuit
        A Qiskit QuantumCircuit object with the specified gates applied to qubit 0,
        followed by a measurement of all qubits.
    """
    qc = QuantumCircuit(len(sequence))
    for gate_num in sequence:
        qc.append(gate_map[int(gate_num)], [0])
    qc.measure_all()
    return qc


def simulate_circuit(gate_sequence, gate_set, qubit_number, shots):
    """Simulate a quantum circuit using a sequence of gates and return the normalized results.

    This function takes a sequence of gates, constructs a quantum circuit for each sequence,
    and then simulates the circuit using Qiskit's Aer qasm simulator. The function collects
    and normalizes the simulation results for analysis.

    Parameters
    ----------
    gate_sequence : list of lists
        A list containing sublists, each of which is a sequence of gate numbers representing
        a quantum circuit.
    gate_set : dict
        A dictionary mapping gate numbers (integers) to Qiskit gate objects.
    qubit_number : int
        The number of qubits in the quantum circuit.
    shots : int
        The number of shots (repetitions) for each circuit simulation.

    Returns
    -------
    numpy.ndarray
        An array of normalized results. Each row in the array corresponds to a gate sequence
        in `gate_sequence`, and each column corresponds to a possible measurement outcome.
        Values are normalized by the total number of shots.
    """
    simulator = Aer.get_backend("qasm_simulator")
    results = []
    dict = {}

    for qubit_number in range(qubit_number + 1):
        bin_value = str(bin(qubit_number)[2:])
        result_name = "0" * (8 - len(bin_value)) + bin_value
        dict[result_name] = []

    for i in gate_sequence:
        qc = get_qiskit_circuits(i, gate_set)
        sequence_result = execute(qc, simulator, shots=shots).result().get_counts()

        for key in dict.keys():
            if key in sequence_result.keys():
                dict[key].append(sequence_result[key])
            else:
                dict[key].append(0)

    for key in dict.keys():
        results.append(dict[key])
    return np.array(results) / shots


def get_gate_sequence(sequence_number, sequence_length, gate_set):
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
    gate_set : list or array
        The set of gates to be used, where each gate is represented by a unique index.

    Returns
    -------
    numpy.ndarray
        An array of shape (sequence_number, sequence_length), where each row represents a
        randomly generated gate sequence.
    """
    J_rand = np.array(random.sample(range(len(gate_set)**sequence_length), sequence_number))
    J = np.array([low_level_jit.local_basis(ind, len(gate_set), sequence_length) for ind in J_rand])
    return J
