import numpy as np
import time

from low_level_jit import *
from additional_fns import *
from optimization import *
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
from scipy.linalg import eig

        

        
def A_B_SFN(K,A,B,y,J,l,d,r,rK,argument = "E"):
    """!

    Do saddle free Newton optimization step on state preparation or measurement

    Parameters
    -------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization  
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    argument : {"E","rho"}
        Argument that is optimized over; Default : "E"


    Returns
    -------
    v_new : numpy array
        Either updated POVM or initial state parametrization, depending on keyword --argument--


    Notes:
        Performs saddle free Newton optimization using the euclidean gradient and Hessian with 
        simple update rule v_new = v + a*delta_v. 
        The step size is determined by minimizing the objective function in the step size parameter.

    """
    derivs = {'rho':ddB_derivs, 'E':ddA_derivs}
    deriv = derivs[argument]
    pdim = int(np.sqrt(r))
    H = np.zeros((2*r,2*r)).astype(np.complex128)
    
    if argument == "rho":
        v = [B]
        C = np.array([(A[i].T.conj()@A[i]).reshape(-1) for i in range(A.shape[0])])
    else:
        v = [A]
        C = (B@B.T.conj()).reshape(-1)
        
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r) 
    dv, dMdM, dMconjdM, dconjdA = deriv(X,K,A,B,J,y,l,d,r,pdim,rK)
    Gamma = np.array([dv.conj(),dv]).reshape(-1) 
    
    
    A00 = dMconjdM + dconjdA
    A10 = dMdM
    A11 = A00.conj()
    A01 = A10.conj()

    H[:r,:r] = A00
    H[:r,r:] = A01
    H[r:,:r] = A10
    H[r:,r:] = A11 
    
    
    evals,U = eigh(H) 
    
    lam = 1e-3
    H_abs_inv = U@np.diag(1/(np.abs(evals) + lam))@U.T.conj()
    delta_v = (H_abs_inv@Gamma)[:r]
    
    
    a = minimize(lineobjf_A_B, 1e-9, args=(v[0],delta_v.reshape(pdim,pdim),X,C,y,J,argument), method = 'COBYLA').x
    v_new = v[0] - a*delta_v.reshape(pdim,pdim)
    
    return v_new


def A_SFN(K,A,B,y,J,l,d,r,rK,n_povm):
    """!

    Do optimization step on POVM parametrization with projected euclidean saddle free Newton step.  

    Parameters
    -------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization  
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements


    Returns
    -------
    A_new : numpy array
        Updated POVM parametrization


    Notes:
        Computes a saddle free Newton update direction based on the eudlidean gradient and Hessian. 
        This update direction is then projected onto the local tangent space and A is updated along the 
        geodesic following the tanget direction.
        The step size is determined by minimizing the objective function in the step size parameter.

    """
    pdim = int(np.sqrt(r))
    rho = (B@B.T.conj()).reshape(-1)
    H = np.zeros((2*r,2*r)).astype(np.complex128)
    delta_A = np.zeros((n_povm,r)).astype(np.complex128)
        
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r) 
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X,K,A,B,J,y,l,d,r,pdim,rK,n_povm)
    Gamma = np.array([np.array([dA_[k].conj(),dA_[k]]).reshape(-1) for k in range(n_povm)]) 
        
    for i in range(n_povm):       

        H[:r,:r] = dMconjdM[i] + dconjdA[i]
        H[r:,:r] = dMdM[i]
        H[:r,r:] = H[r:,:r].conj()
        H[r:,r:] = H[:r,:r].conj()

        evals,U = eigh(H) 
 
        lam = 1e-3
        H_abs_inv = U@np.diag(1/(np.abs(evals) + lam))@U.T.conj()
        delta_A[i] = (H_abs_inv@Gamma[i])[:r]
    G = tangent_proj(A,delta_A.reshape(n_povm*pdim,pdim),1,n_povm)[0]
    
    a = minimize(lineobjf_A_geodesic, 1e-9, args=(G,X,A,rho,J,y), method = 'COBYLA').x
    A_new = update_A_geodesic(A,G,a)
    
    return A_new

def A_SFN_riem_Hess(K,A,B,y,J,l,d,r,rK,n_povm,lam = 1e-3):
    pdim = int(np.sqrt(r))
    n = n_povm*pdim
    nt = n_povm*r
    rho = (B@B.T.conj()).reshape(-1)
    H = np.zeros((2,nt,2,nt)).astype(np.complex128)
    P_T = np.zeros((2,nt,2,nt)).astype(np.complex128)
    Fyconjy = np.zeros((n_povm,r,n_povm,r)).astype(np.complex128)
    Fyy = np.zeros((n_povm,r,n_povm,r)).astype(np.complex128)
        
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r) 
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X,K,A,B,J,y,l,d,r,pdim,rK,n_povm)
    
    #Second derivatives
    for i in range(n_povm):   
        Fyconjy[i,:,i,:] = dMconjdM[i] + dconjdA[i]
        Fyy[i,:,i,:] = dMdM[i]

    #derivative
    Fy = dA_.reshape(n,pdim)
    Y = A.reshape(n,pdim)
    rGrad = Fy.conj() - Y@Fy.T@Y 
    G = np.array([rGrad,rGrad.conj()]).reshape(-1)

    P = np.eye(n) - Y@Y.T.conj() 
    T = transp(n,pdim)
    
    H00 = -(np.kron(Y,Y.T))@T@Fyy.reshape(nt,nt).T + Fyconjy.reshape(nt,nt).T.conj() -(np.kron(np.eye(n),Y.T@Fy))/2 - (np.kron(Y@Fy.T,np.eye(pdim)))/2 -(np.kron(P,Fy.T.conj()@Y.conj()))/2       
    H01 = Fyy.reshape(nt,nt).T.conj() - np.kron(Y,Y.T)@T@Fyconjy.reshape(nt,nt).T + (np.kron(Fy.conj(),Y.T)@T)/2 + (np.kron(Y,Fy.T.conj())@T)/2

    H[0,:,0,:] = H00
    H[0,:,1,:] = H01 
    H[1,:,0,:] = H01.conj()
    H[1,:,1,:] = H00.conj()        

    P_T[0,:,0,:] = np.eye(nt) - np.kron(Y@Y.T.conj(),np.eye(pdim))/2
    P_T[0,:,1,:] = - np.kron(Y,Y.T)@T/2
    P_T[1,:,0,:] = P_T[0,:,1,:].conj()
    P_T[1,:,1,:] = P_T[0,:,0,:].conj() 

    H = H.reshape(2*nt,2*nt)@P_T.reshape(2*nt,2*nt)

    #saddle free newton method 
    H = (H + H.T.conj())/2
    evals,U = eigh(H)    
    H_abs_inv = U@np.diag(1/(np.abs(evals) + lam))@U.T.conj()
    
    Delta_A = ((H_abs_inv@G)[:nt]).reshape(n,pdim)
        
    Delta = tangent_proj(A,Delta_A,1,n_povm)[0]
    
    a = minimize(lineobjf_A_geodesic, 1e-9, args=(Delta,X,A,rho,J,y), method = 'COBYLA').x
    
    return update_A_geodesic(A,Delta,a)


def gd(K,E,rho,y,J,l,d,r,rK, ls = 'COBYLA'):
    """!

    Do gradient descent optimization step on gates

    Parameters
    -------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize
    }


    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations


    Notes:
        Gradient descent using the Riemannian gradient and updating along the geodesic. 
        The step size is determined by minimizing the objective function in the step size parameter. 

    """    
    #setup
    pdim = int(np.sqrt(r))
    n = rK*pdim
    nt = rK*r
    Delta = np.zeros((d,n,pdim)).astype(np.complex128)
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r)
    
    dK_ = dK(X,K,E,rho,J,y,l,d,r,rK)  
    for k in range(d):        
        #derivative
        Fy = dK_[k].reshape(n,pdim)
        Y = K[k].reshape(n,pdim)
        rGrad = Fy.conj() - Y@Fy.T@Y #Riem. gradient taken from conjugate derivative
        Delta[k] = rGrad  
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta,K,E,rho,J,y), method = ls, options={'maxiter':200})
    a = res.x  
    K_new = update_K_geodesic(K,Delta,a)
        
    return K_new



def SFN_riem_Hess(K,E,rho,y,J,l,d,r,rK,lam = 1e-3, ls = 'COBYLA'):
    """!

    Riemannian saddle free Newton step on each gate individually

    Parameters
    -------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize
    }


    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations

    """    
    #setup
    pdim = int(np.sqrt(r))
    n = rK*pdim
    nt = rK*r
    H = np.zeros((2*nt,2*nt)).astype(np.complex128)
    P_T = np.zeros((2*nt,2*nt)).astype(np.complex128)
    Delta_K = np.zeros((d,rK,pdim,pdim)).astype(np.complex128)
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r)
    
    #compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X,K,E,rho,J,y,l,d,r,rK)
    dd, dconjd = ddM(X,K,E,rho,J,y,l,d,r,rK)

    #Second derivatives
    Fyconjy = dM11.reshape(d,nt,d,nt) + np.einsum('ijklmnop->ikmojlnp',dconjd).reshape(d,nt,d,nt)
    Fyy = dM10.reshape(d,nt,d,nt) + np.einsum('ijklmnop->ikmojlnp',dd).reshape(d,nt,d,nt)
    
    for k in range(d):        

        Fy = dK_[k].reshape(n,pdim)
        Y = K[k].reshape(n,pdim)
        rGrad = Fy.conj() - Y@Fy.T@Y #riemannian gradient, taken from conjugate derivative
        G = np.array([rGrad,rGrad.conj()]).reshape(-1)
        
        P = np.eye(n) - Y@Y.T.conj() 
        T = transp(n,pdim)

        #Riemannian Hessian with correction terms
        H00 = -(np.kron(Y,Y.T))@T@Fyy[k,:,k,:].T + Fyconjy[k,:,k,:].T.conj() -(np.kron(np.eye(n),Y.T@Fy))/2 - (np.kron(Y@Fy.T,np.eye(pdim)))/2 -(np.kron(P,Fy.T.conj()@Y.conj()))/2       
        H01 = Fyy[k,:,k,:].T.conj() - np.kron(Y,Y.T)@T@Fyconjy[k,:,k,:].T + (np.kron(Fy.conj(),Y.T)@T)/2 + (np.kron(Y,Fy.T.conj())@T)/2
        
                
        H[:nt,:nt] = H00
        H[:nt,nt:] = H01 
        H[nt:,:nt] = H[:nt,nt:].conj()
        H[nt:,nt:] = H[:nt,:nt].conj()        
        

        
        
        #Tangent space projection
        P_T[:nt,:nt] = np.eye(nt)- np.kron(Y@Y.T.conj(),np.eye(pdim))/2
        P_T[:nt,nt:] = - np.kron(Y,Y.T)@T/2
        P_T[nt:,:nt] = P_T[:nt,nt:].conj()
        P_T[nt:,nt:] = P_T[:nt,:nt].conj()  
        
        H = H@P_T
        
        #saddle free newton method 
        evals,S = eig(H) 
        
        H_abs_inv = S@np.diag(1/(np.abs(evals)+ lam))@la.inv(S)
        Delta_K[k] = ((H_abs_inv@G)[:nt]).reshape(rK,pdim,pdim)
    
    Delta = tangent_proj(K,Delta_K,d,rK)        
    
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta,K,E,rho,J,y), method = ls, options={'maxiter':20})
    a = res.x    
    K_new = update_K_geodesic(K,Delta,a), np.linalg.norm(Delta_K)
        
    return K_new



def SFN_riem_Hess_full(K,E,rho,y,J,l,d,r,rK,lam = 1e-3, ls = 'COBYLA'):
    """!

    Riemannian saddle free Newton step on product manifold of all gates

    Parameters
    -------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize
    }


    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations

    """ 
    
    #setup
    pdim = int(np.sqrt(r))
    n = rK*pdim
    nt = rK*r
    H = np.zeros((2,d,nt,2,d,nt)).astype(np.complex128)
    P_T = np.zeros((2,d,nt,2,d,nt)).astype(np.complex128)
    G = np.zeros((2,d,nt)).astype(np.complex128)
    X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r)
    
    #compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X,K,E,rho,J,y,l,d,r,rK)
    dd, dconjd = ddM(X,K,E,rho,J,y,l,d,r,rK)

    #Second derivatives
    Fyconjy = dM11.reshape(d,nt,d,nt) + np.einsum('ijklmnop->ikmojlnp',dconjd).reshape(d,nt,d,nt)
    Fyy = dM10.reshape(d,nt,d,nt) + np.einsum('ijklmnop->ikmojlnp',dd).reshape(d,nt,d,nt)

    for k in range(d):        
        Fy = dK_[k].reshape(n,pdim)
        Y = K[k].reshape(n,pdim)
        rGrad = Fy.conj() - Y@Fy.T@Y 
        
        G[0,k,:] = rGrad.reshape(-1)
        G[1,k,:] = rGrad.conj().reshape(-1)
        
        P = np.eye(n) - Y@Y.T.conj() 
        T = transp(n,pdim)
        H00 = -(np.kron(Y,Y.T))@T@Fyy[k,:,k,:].T + Fyconjy[k,:,k,:].T.conj() -(np.kron(np.eye(n),Y.T@Fy))/2 - (np.kron(Y@Fy.T,np.eye(pdim)))/2 -(np.kron(P,Fy.T.conj()@Y.conj()))/2       
        H01 = Fyy[k,:,k,:].T.conj() - np.kron(Y,Y.T)@T@Fyconjy[k,:,k,:].T + (np.kron(Fy.conj(),Y.T)@T)/2 + (np.kron(Y,Fy.T.conj())@T)/2  
        
        #Riemannian Hessian with correction terms
        H[0,k,:,0,k,:] = H00 
        H[0,k,:,1,k,:] = H01
        H[1,k,:,0,k,:] = H01.conj()
        H[1,k,:,1,k,:] = H00.conj()
        
        #Tangent space projection
        P_T[0,k,:,0,k,:] = np.eye(nt) - np.kron(Y@Y.T.conj(),np.eye(pdim))/2
        P_T[0,k,:,1,k,:] = - np.kron(Y,Y.T)@T/2
        P_T[1,k,:,0,k,:] = P_T[0,k,:,1,k,:].conj()
        P_T[1,k,:,1,k,:] = P_T[0,k,:,0,k,:].conj()
        

        
        for k2 in range(d):
            if k2 != k:
                Yk2 = K[k2].reshape(n,pdim)
                H[0,k2,:,0,k,:] = Fyconjy[k,:,k2,:].T.conj()-np.kron(Yk2,Yk2.T)@T@Fyy[k,:,k2,:].T
                H[0,k2,:,1,k,:] = Fyy[k,:,k2,:].T.conj()-np.kron(Yk2,Yk2.T)@T@Fyconjy[k,:,k2,:].T
                H[1,k2,:,0,k,:] = H[0,k2,:,1,k,:].conj()
                H[1,k2,:,1,k,:] = H[0,k2,:,0,k,:].conj()
             
            
    H = H.reshape(2*d*nt,-1)@P_T.reshape(2*d*nt,-1)
    
    #application of saddle free newton method
    H = (H + H.T.conj())/2
    evals,U = eigh(H)    
    H_abs_inv = U@np.diag(1/(np.abs(evals) + lam))@U.T.conj()
    Delta_K = ((H_abs_inv@G.reshape(-1))[:d*nt]).reshape(d,rK,pdim,pdim)

    Delta = tangent_proj(K,Delta_K,d,rK) #Delta_K is already in tangent space but not to sufficient numerical accuracy
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta,K,E,rho,J,y), method = ls, options={'maxiter':20})
    a = res.x
    K_new = update_K_geodesic(K,Delta,a)

    return K_new


def optimize(y,J,l,d,r,rK,n_povm, method, K, E, rho, A, B):
    """!

    Full gate set optimization update alternating on E, K and rho

    Parameters
    -------
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    K : numpy array
        Current estimates of Kraus operators
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    A : numpy array
        Current POVM parametrization  
    B : numpy array
        Current initial state parametrization
    }


    Returns
    -------
    K_new : numpy array
        Updated estimates of Kraus operators
    X_new : numpy array
        Updated estimates of superoperatos corresponding to K_new
    E_new : numpy array
        Updated POVM estimate
    rho_new : numpy array
        Updated initial state estimate
    A_new : numpy array
        Updated POVM parametrization  
    B_new : numpy array
        Updated initial state parametrization

    """ 
    pdim = int(np.sqrt(r))
    A_new = A_SFN_riem_Hess(K,A,B,y,J,l,d,r,rK,n_povm)
    E_new = np.array([(A_new[i].T.conj()@A_new[i]).reshape(-1) for i in range(n_povm)])
    if method == 'SFN':
        K_new = SFN_riem_Hess_full(K,E_new,rho,y,J,l,d,r,rK,lam = 1e-3, ls = 'COBYLA')
    elif method == 'GD':
        K_new = gd(K,E_new,rho,y,J,l,d,r,rK, ls = 'COBYLA')
    B_new = A_B_SFN(K_new,A_new,B,y,J,l,d,r,rK,argument = "rho")
    rho_new = (B_new@B_new.T.conj()).reshape(-1)
    rho_new = rho_new/np.trace(rho_new.reshape(pdim,pdim))
    X_new = np.einsum('ijkl,ijnm -> iknlm', K_new, K_new.conj()).reshape(d,r,r)
    return K_new, X_new, E_new, rho_new, A_new, B_new


def run_mGST(*args, method = 'SFN', max_inits = 10, 
             max_iter = 200, final_iter = 70, target_rel_prec = 1e-4, 
             init = []):
    """!

    Main mGST routine

    Parameters
    -------
    y : numpy array
        2D array of measurement outcomes for sequences in J; 
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array 
        2D array where each row contains the gate indices of a gate sequence  
    l : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    bsize : int
        Size of the batch (number of sequences)
    meas_samples : int
        Number of samples taken per gate sequence to obtain measurement array y
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    max_reruns : int
        Maximum number or reinitializations; Default: 10
    max_iter : int
        Maximum number of iterations on batches; Default: 200
    final_iter : int
        Maximum number of iterations on full data set; Default: 70
    target_rel_prec : float
        Target precision relative to stopping value at which the final iteration loop breaks
    init : [ , , ]
        List of 3 numpy arrays in the format [X,E,rho], that can be used as an initialization;
        If no initialization is given a random initialization is used


    Returns
    -------
    K : numpy array
        Updated estimates of Kraus operators
    X : numpy array
        Updated estimates of superoperatos corresponding to K_new
    E : numpy array
        Updated POVM estimate
    rho : numpy array
        Updated initial state estimate
    res_list : list
        Collected objective function values after each iteration

    """ 
    y,J,l,d,r,rK,n_povm, bsize, meas_samples = args
    t0 = time.time()
    pdim = int(np.sqrt(r))
    delta = 3*(1-y.reshape(-1))@y.reshape(-1)/len(J)/n_povm/meas_samples
    
    if init:
        K = init[0]
        E = init[1] 
        rho = init[2]+1e-14*np.eye(pdim).reshape(-1) #offset small negative eigenvalues for stability
        A = np.array([la.cholesky(E[k].reshape(pdim,pdim)+1e-14*np.eye(pdim)).T.conj()
                      for k in range(n_povm)]) 
        B = la.cholesky(rho.reshape(pdim,pdim))
        X = np.einsum('ijkl,ijnm -> iknlm', K, K.conj()).reshape(d,r,r)   
        max_reruns = 1

    succ = 0
    for i in range(max_inits):
        if not init:
            K,X,E,rho = random_gs(d,r,rK,n_povm)
            A = np.array([la.cholesky(E[k].reshape(pdim,pdim)+1e-14*np.eye(pdim)).T.conj()
                      for k in range(n_povm)]) 
            B = la.cholesky(rho.reshape(pdim,pdim))
        res_list = [objf(X,E,rho,J,y,d,l)]
        for j in range(max_iter):
            yb,Jb = batch(y,J,bsize)
            K,X,E,rho,A,B = optimize(yb,Jb,l,d,r,rK, n_povm, method, K, E, rho, A, B)
            res_list.append(objf(X,E,rho,J,y,d,l)) 
            if res_list[-1] < delta:
                succ = 1
                print('Initialization successful, improving estimate over full data....')
                break  
        if succ == 1:
            break 
        if i+1 < max_inits:
            print('Run ', i, 'failed, trying new initialization...')
        else:
            print('Maximum number of reinitializations reached without landing below success threshold, attempting optimization over full data set...')
    for n in range(final_iter):
        K,X,E,rho,A,B = optimize(y,J,l,d,r,rK,
                                          n_povm, method, K, E, rho, A, B)
        res_list.append(objf(X,E,rho,J,y,d,l))
        if np.abs(res_list[-2]-res_list[-1])<delta*target_rel_prec:
            break
    print('#################')
    if succ == 1 or (res_list[-1] < delta):
        print('\t Convergence crit. satisfied')
    else: 
        print('\t Convergence crit. not satisfied')
    print('\t Final objective function value',res_list[-1],
          'with # of initializations: %i'%(i+1),
         '\n \t Total runtime:',time.time()-t0)
    return K, X, E, rho, res_list