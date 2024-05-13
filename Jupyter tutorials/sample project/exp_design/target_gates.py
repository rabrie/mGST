import numpy as np

pdim = 2   # physical dimension
r = pdim**2   # rank of the gate superoperators 
d = 6 # number of different gates
n_povm = 2

Id = np.array([[1,0],[0,1]]).astype(np.complex128)
Rx = np.array([[0,1],[1,0]]).astype(np.complex128)
Ry = np.array([[0,-1j],[1j,0]]).astype(np.complex128)
Rx_half = (1/np.sqrt(2))*np.array([[1,1j],[1j,1]]).astype(np.complex128)
Ry_half = (1/np.sqrt(2))*np.array([[1,1],[-1,1]]).astype(np.complex128)

K_true = np.zeros((d,4,pdim,pdim)).astype(np.complex128)
K_true[0,0] = Id
K_true[1,0] = Id
K_true[2,0] = Rx
K_true[3,0] = Ry
K_true[4,0] = Rx_half
K_true[5,0] = Ry_half
X_true = np.einsum('ijkl,ijnm -> iknlm', K_true, K_true.conj()).reshape(d,r,r)  

E_true = np.zeros((n_povm,r)).astype(np.complex128)
for i in range(n_povm):
    basis_vec = np.zeros(pdim)
    basis_vec[i] = 1
    E_true[i,:] = np.outer(basis_vec.conj(), basis_vec).reshape(r)
rho_true = E_true[0]