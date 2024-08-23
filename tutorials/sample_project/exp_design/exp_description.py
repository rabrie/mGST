import numpy as np
from qiskit.circuit.library import IGate, RGate
from mGST.qiskit_interface import qiskit_gate_to_kraus, add_idle_gates, remove_idle_wires
from mGST.additional_fns import basis
from qiskit import QuantumCircuit
from qiskit import QuantumRegister


# The qubits on which GST-experiments are run
active_qubits = [0]

# physical dimension of the subsystem of interest
pdim = 2**len(active_qubits)

# Number of distinct gates in the gate set
d = 5

# Number of distinct measurement outcomes in the data. For computational basis measurements n_povm = pdim.
n_povm = pdim

# Kraus rank of the reconstruction.
rK = 4

#Number of shots per sequence
meas_samples = 1000


# Initialization
# Boolean variable determines whether or not the target gate set is used as initialization.
# Setting this to 'true' can greatly speed up the algorithm, but it can lead to wrong results when the
# real gates are not close to the target gates (either due to exp. errors or wrong target gate assignment below).
from_init = True

# Optimization method: Options are gradient descend: "GD" or saddle free Newton: "SFN".
# SFN is recommended for single qubits GST and two qubits with rK < 4.
# "GD" is recommended for 2 qubits with rK >= 4 and 3 qubits with any Kraus rank.
opt_method = "SFN"

experiment_name = 'test_experiment' #Short name to identify the experiment. This will be used as a filename for GST results.
experiment_date = '19.07.2024'
folder_name = "exp_data" #The name of the folder where measurement data is stored

basis_dict = {'0':0, '1':1} # Translating from the povm outcomes labels of the data file to numbers from 0 to n_povm
gate_labels = {0: "Idle",  1: "Rx(pi)", 2: "Ry(pi)", 3: "Rx(pi/2)", 4: "Ry(pi/2)"}


gate_set = [QuantumCircuit(len(active_qubits),0) for _ in range(d)]
gate_list = [IGate(),
             RGate(np.pi, 0), RGate(np.pi, np.pi/2),
             RGate(0.5*np.pi, 0), RGate(0.5*np.pi, np.pi/2)]
gate_qubits = [[0], [0], [0], [0], [0]]

for i in range(d):
    gate_set[i].append(gate_list[i], gate_qubits[i])

# Transforming the single gate circuits to Operators
K_target = qiskit_gate_to_kraus(gate_set)
X_target = np.einsum('ijkl,ijnm -> iknlm', K_target, K_target.conj()
                   ).reshape(d, pdim**2, pdim**2)   # tensor of superoperators

# Initial state |0>
rho_target = np.kron(basis(pdim,0).T.conj(), basis(pdim,0)).reshape(-1).astype(np.complex128)

# Computational basis measurement:
E_target = np.array(
    [np.kron(basis(pdim,i).T.conj(), basis(pdim,i)).reshape(-1)
                     for i in range(pdim)]
    ).astype(np.complex128)






