from numba import jit,njit,prange
import numpy as np
from math import pi
import os


#way to delete cached functions for use on different machine
def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)

@njit(cache=True)
def local_basis(x, b, l):
    """
    Converts given number x from base 10 to base b and returns new digits 
    in array of length l with leading zeros
    x -- input number in base 10
    b -- base to convert to
    l -- number of output digits in base b
    """
    r = np.zeros(l).astype(np.int32)
    k = 1
    while x > 0:
        r[-k] = x % b
        x //= b
        k += 1
    return r

@njit(cache=True)
def contract(X,j_vec):
    j_vec = j_vec[j_vec>=0]
    res = np.eye(X[0].shape[0])
    res = res.astype(np.complex128)
    for j in j_vec:
        res = res.dot(X[j])
    return res

@njit(cache=True, fastmath = True)
def objf(X,E,rho,J,y,d,l):
    m = len(J)
    n_povm = y.shape[0]
    objf = 0
    for i in prange(m):
        j = J[i][J[i]>=0]
        C = contract(X,j)
        for o in range(n_povm):
            objf += abs(E[o].conj()@C@rho - y[o,i])**2
    return objf/m/n_povm


@njit(cache = True)
def MVE_lower(X_true,E_true,rho_true,X,E,rho,J,d,l,n_povm): 
    m = len(J)
    dist :float = 0
    max_dist :float = 0
    curr :float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true,j)
        C = contract(X,j)
        curr = 0
        for k in range(n_povm):
            y_t = E_true[k].conj()@C_t@rho_true
            y = E[k].conj()@C@rho
            curr += np.abs(y_t - y)
        curr = curr/2
        dist += curr
        if curr > max_dist:
            max_dist = curr
    return dist/m, max_dist


@njit(cache = True)
def Mp_norm_lower(X_true,E_true,rho_true,X,E,rho,J,d,l,n_povm,p): 
    m = len(J)
    dist :float = 0
    max_dist :float = 0
    curr :float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true,j)
        C = contract(X,j)
        for k in range(n_povm):
            y_t = E_true[k].conj()@C_t@rho_true
            y = E[k].conj()@C@rho
            dist += np.abs(y_t - y)**p
        if curr > max_dist:
            max_dist = curr
    return dist**(1/p)/m/n_povm, max_dist**(1/p)


@njit(cache=True)
def dK(X,K,E,rho,J,y,l,d,r,rK):
    K = K.reshape(d,rK,-1)
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    dK = np.zeros((d,rK,r))
    dK = np.ascontiguousarray(dK.astype(np.complex128))
    m = len(J)
    for k in range(d):
        for n in range(m):
            j = J[n][J[n]>=0]
            for i in range(len(j)):
                if j[i] == k:
                    for o in range(n_povm):
                        L = E[o].conj()@contract(X,j[:i])
                        R = contract(X,j[i+1:])@rho
                        D_ind = (L@X[k]@R-y[o,n]) 
                        dK[k] += D_ind*K[k].conj()@np.kron(L.reshape(pdim,pdim).T,R.reshape(pdim,pdim).T)
    return dK.reshape(d,rK,pdim,pdim)*2/m/n_povm

@njit(cache = True)
def dK_dMdM(X,K,E,rho,J,y,l,d,r,rK):
    K = K.reshape(d,rK,-1)
    pdim = int(np.sqrt(r))
    n = d*rK*r
    n_povm = y.shape[0]
    dK = np.zeros((d,rK,r)).astype(np.complex128)
    dM11 = np.zeros(n**2).astype(np.complex128)
    dM10 = np.zeros(n**2).astype(np.complex128)
    m = len(J)
    for n in range(m):
        j = J[n][J[n]>=0]
        dM = np.ascontiguousarray(np.zeros((n_povm,d,rK,r)).astype(np.complex128))
        for i in range(len(j)):
            k = j[i]
            C = contract(X,j[:i])
            R = contract(X,j[i+1:])@rho
            for o in range(n_povm):
                L = E[o].conj()@C
                D_ind = L@X[k]@R-y[o,n] 
                dM_loc = K[k].conj()@np.kron(L.reshape(pdim,pdim).T,R.reshape(pdim,pdim).T)
                dM[o,k,:,:] += dM_loc
                dK[k] += D_ind*dM_loc
        for o in range(n_povm):
            dM11 += np.kron(dM[o].conj().reshape(-1),dM[o].reshape(-1))
            dM10 += np.kron(dM[o].reshape(-1),dM[o].reshape(-1))
    return dK.reshape(d,rK,pdim,pdim)*2/m/n_povm, 2*dM10/m/n_povm, 2*dM11/m/n_povm


@njit(cache = True,parallel = True) 
def ddM(X,K,E,rho,J,y,l,d,r,rK):
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    ddK = np.zeros((d**2,rK**2,r,r))
    ddK = np.ascontiguousarray(ddK.astype(np.complex128))
    dconjdK = np.zeros((d**2,rK**2,r,r))
    dconjdK = np.ascontiguousarray(ddK.astype(np.complex128))
    m = len(J)
    for k in prange(d**2):
        k1,k2 = local_basis(k,d,2)
        for n in range(m):
            j = J[n][J[n]>=0]
            for i1 in range(len(j)):
                if j[i1] == k1:
                    for i2 in range(len(j)):
                        if j[i2] == k2:
                            L0 = contract(X,j[:min(i1,i2)])
                            C = contract(X,j[min(i1,i2)+1:max(i1,i2)]).reshape(pdim,pdim,pdim,pdim)
                            R = contract(X,j[max(i1,i2)+1:])@rho
                            for o in range(n_povm):
                                L = E[o].conj()@L0
                                if i1 == i2:
                                    D_ind = L@X[k1]@R-y[o,n]
                                elif i1 < i2:
                                    D_ind = L@X[k1]@C.reshape(r,r)@X[k2]@R-y[o,n]
                                elif i1 > i2:
                                    D_ind = L@X[k2]@C.reshape(r,r)@X[k1]@R-y[o,n]  

                                ddK_loc = np.zeros((rK**2,r,r)).astype(np.complex128)
                                dconjdK_loc = np.zeros((rK**2,r,r)).astype(np.complex128)
                                for rk1 in range(rK):
                                    for rk2 in range(rK):
                                        if i1 < i2:
                                            ddK_loc[rk1*rK+rk2] = np.kron(L.reshape(pdim,pdim)@K[k1,rk1].conj(),R.reshape(pdim,pdim)@K[k2,rk2].T.conj())@np.ascontiguousarray(C.transpose(1,3,0,2)).reshape(r,r)
                                            ddK_loc[rk1*rK+rk2] = np.ascontiguousarray(ddK_loc[rk1*rK+rk2].reshape(pdim,pdim,pdim,pdim).transpose(0,3,2,1)).reshape(r,r)

                                            dconjdK_loc[rk1*rK+rk2] = np.kron(L.reshape(pdim,pdim)@K[k1,rk1].conj(),R.reshape(pdim,pdim).T@K[k2,rk2].T)@np.ascontiguousarray(C.transpose(1,2,3,0)).reshape(r,r)
                                            dconjdK_loc[rk1*rK+rk2] = np.ascontiguousarray(dconjdK_loc[rk1*rK+rk2].reshape(pdim,pdim,pdim,pdim).transpose(0,2,3,1)).reshape(r,r)
                                        elif i1 == i2:
                                            dconjdK_loc[rk1*rK+rk2] = np.outer(L,R)
                                        elif i1 > i2:
                                            ddK_loc[rk1*rK+rk2] = np.kron(L.reshape(pdim,pdim)@K[k2,rk2].conj(),R.reshape(pdim,pdim)@K[k1,rk1].T.conj())@np.ascontiguousarray(C.transpose(1,3,0,2)).reshape(r,r)
                                            ddK_loc[rk1*rK+rk2] = np.ascontiguousarray(ddK_loc[rk1*rK+rk2].reshape(pdim,pdim,pdim,pdim).transpose(3,0,1,2)).reshape(r,r)

                                            dconjdK_loc[rk1*rK+rk2] = np.kron(L.reshape(pdim,pdim).T@K[k2,rk2],R.reshape(pdim,pdim)@K[k1,rk1].T.conj())@np.ascontiguousarray(C.transpose((0,3,2,1))).reshape(r,r)
                                            dconjdK_loc[rk1*rK+rk2] = np.ascontiguousarray(dconjdK_loc[rk1*rK+rk2].reshape(pdim,pdim,pdim,pdim).transpose(2,0,1,3)).reshape(r,r)                                      
                                ddK[k1*d + k2] += D_ind*ddK_loc
                                dconjdK[k1*d + k2] += D_ind*dconjdK_loc
    return ddK.reshape(d,d,rK,rK,pdim,pdim,pdim,pdim)*2/m/n_povm, dconjdK.reshape(d,d,rK,rK,pdim,pdim,pdim,pdim)*2/m/n_povm

@njit(parallel=True,cache=True)
def dA(X,K,A,B,J,y,l,d,r,pdim,rK,n_povm):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm,r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj()@A[k]).reshape(-1)
    rho = (B@B.T.conj()).reshape(-1)
    dA = np.zeros((n_povm,pdim,pdim))
    dA = dA.astype(np.complex128)
    m = len(J)
    for n in prange(m):
        jE = J[n][J[n]>=0][0]
        j = J[n][J[n]>=0][1:]
        inner_deriv = contract(X,j)@rho
        D_ind = E[jE].conj().dot(inner_deriv)-y[n]
        dA[jE] += D_ind*A[jE]@inner_deriv.reshape(pdim,pdim).T.conj()
    return dA

@njit(parallel=True,cache=True)
def dB(X,K,A,B,J,y,l,d,r,pdim,rK):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = (A.T.conj()@A).reshape(-1)
    rho = (B@B.T.conj()).reshape(-1)
    dB = np.zeros((pdim,pdim))
    dB = dB.astype(np.complex128)
    m = len(J)
    for n in prange(m):
        jE = J[n][J[n]>=0][0]
        j = J[n][J[n]>=0][1:]
        inner_deriv = E[jE].conj().dot(contract(X,j))
        D_ind = inner_deriv.dot(rho)-y[n]
        dB += D_ind*inner_deriv.reshape(pdim,pdim).conj()@B
    return dB

@njit(parallel=True,cache=True)
def ddA_derivs(X,K,A,B,J,y,l,d,r,pdim,rK,n_povm):
    #Derivatives depend only on one povm element, different povms elements are only connected via isometry condition    
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm,r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj()@A[k]).reshape(-1)
    rho = (B@B.T.conj()).reshape(-1)
    dA = np.zeros((n_povm,pdim,pdim)).astype(np.complex128)
    dM = np.zeros((pdim,pdim)).astype(np.complex128)
    dMdM = np.zeros((n_povm,r,r)).astype(np.complex128)
    dMconjdM = np.zeros((n_povm,r,r)).astype(np.complex128)
    dconjdA = np.zeros((n_povm,r,r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):
        j = J[n][J[n]>=0]
        R = contract(X,j)@rho
        for o in range(n_povm): 
            D_ind = E[o].conj()@R-y[o,n]
            dM = (A[o].conj()@R.reshape(pdim,pdim).T)
            dMdM[o] += np.outer(dM,dM)
            dMconjdM[o] += np.outer(dM.conj(),dM)
            dA[o] += D_ind*dM
            dconjdA[o] += D_ind*np.kron(np.eye(pdim).astype(np.complex128),R.reshape(pdim,pdim).T)
    return dA*2/m/n_povm, dMdM*2/m/n_povm, dMconjdM*2/m/n_povm, dconjdA*2/m/n_povm

@njit(parallel=True,cache=True)
def ddB_derivs(X,K,A,B,J,y,l,d,r,pdim,rK):
    n_povm = A.shape[0]
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm,r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj()@A[k]).reshape(-1)
    rho = (B@B.T.conj()).reshape(-1)
    dB = np.zeros((pdim,pdim)).astype(np.complex128)
    dM = np.zeros((pdim,pdim)).astype(np.complex128)
    dMdM = np.zeros((r,r)).astype(np.complex128)
    dMconjdM = np.zeros((r,r)).astype(np.complex128)
    dconjdB = np.zeros((r,r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):
        j = J[n][J[n]>=0]
        C = contract(X,j)
        for o in range(n_povm):
            L = E[o].conj()@C
            D_ind = L@rho-y[o,n]

            dM = L.reshape(pdim,pdim)@B.conj()
            dMdM += np.outer(dM,dM)
            dMconjdM += np.outer(dM.conj(),dM)

            dB += D_ind*dM
            dconjdB += D_ind*np.kron(L.reshape(pdim,pdim),np.eye(pdim).astype(np.complex128))
    return dB*2/m/n_povm, dMdM*2/m/n_povm, dMconjdM*2/m/n_povm, dconjdB.T*2/m/n_povm
