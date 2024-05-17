import numpy as np

pdim = 2   # physical dimension

# Number of distinct gates in the gate set
d = 6 

# Number of distinct measurement outcomes in the data. For copmutational basis measurements n_povm = pdim.
n_povm = 2 

# Kraus rank of the reconstruction. 
rK = 1 


# Initialization 
# Boolean variable determines whether or not the target gate set is used as initialization. 
# Setting this to 'true' can greatly speed up the algorithm, but it can lead to wrong results when the
# real gates are not close to the target gates (either due to exp. erorrs or wrong target gate assignment below). 
from_init = True 

experiment_name = 'test' #Short name to identify the experiment. This will be used as a filename for GST results. 
experiment_date = '17.01.2024'
folder_name = "exp_data" #The name of the folder where measurement data is stored
basis_dict = {'00':0, '01':0, '10':1, '11':1} # Translating from the povm outcomes labels of the data file to numbers from 0 to n_povm
gate_labels = {0: "Idle-short", 1: "Idle-long", 2: "Rx(pi)", 3: "Ry(pi)", 4: "Rx(pi/2)", 5: "Ry(pi/2)"}

# ## Example definition for target gates (must be numpy arrays and in computational basis)
Id = np.array([[1,0],[0,1]])
Rx = np.array([[0,1],[1,0]])
Ry = np.array([[0,-1j],[1j,0]])
Rx_half = (1/np.sqrt(2))*np.array([[1,1j],[1j,1]])
Ry_half = (1/np.sqrt(2))*np.array([[1,1],[-1,1]])

K_target = np.zeros((d,4,pdim,pdim)).astype(np.complex128)
K_target[0,0] = Id
K_target[1,0] = Id
K_target[2,0] = Rx
K_target[3,0] = Ry
K_target[4,0] = Rx_half
K_target[5,0] = Ry_half
X_target = np.einsum('ijkl,ijnm -> iknlm', K_target, K_target.conj()).reshape(d,pdim**2,pdim**2)  

E_target = np.zeros((n_povm,pdim**2)).astype(np.complex128)
for i in range(n_povm):
    basis_vec = np.zeros(pdim)
    basis_vec[i] = 1
    E_target[i,:] = np.outer(basis_vec.conj(), basis_vec).reshape(pdim**2)
rho_target = E_target[0]



