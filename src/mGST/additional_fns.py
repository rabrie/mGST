import random
import warnings
import numpy as np
import numpy.linalg as la
from scipy.linalg import qr, expm
from mGST.low_level_jit import local_basis, MVE_lower, Mp_norm_lower, contract, local_basis


def transp(dim1, dim2):
    """Superoperator of a map that performs the transpose operation

    Parameters
    ----------
    dim1 : int
        First dimension of the matrix to be transposed
    dim2 : int
        Second dimension of the matrix to be transposed

    Returns
    -------
    T: numpy array
        Transpose superoperator that turns a dim1xdim2 matrix into a dim2xdim1 matrix

    Notes:
        The projection is done with respect to the canonical metrik.
    """
    Id1 = np.eye(dim1)
    Id2 = np.eye(dim2)
    T = np.einsum("il,jk->ijkl", Id2, Id1).reshape(dim1 * dim2, dim1 * dim2)
    return T


def randpsd(n, normalized="True"):
    """Generate a random positive semidefinite square matrix

    Parameters
    ----------
    n : int
        Number of matrix entries
    normalized : {True, False}, optional
        Controls if the output is trace normalized, defaults to "True"

    Returns
    -------
    mat: numpy array
        Random positive definite square matrix

    Notes:
        Eigenvalues are drawn uniformly from the interval [0,1);
        The basis is generated by diagonalizing a random hermitian matrix (see randHerm function).
    """
    dim = int(np.sqrt(n))
    H = randHerm(dim)
    U, _, _ = np.linalg.svd(H)
    evals = np.random.random_sample(dim)
    if normalized == "True":
        evals /= np.sum(evals)
    mat = np.dot(np.dot(U, np.diag(evals)), U.T.conj())
    mat = mat.reshape(-1)
    return mat


def randvec(n):
    """Generate vector with real and imaginary part drawn from the normal distribution

    Parameters
    ----------
    n : int
        Number of elements for the random vector

    Returns
    -------
    g: 1D numpy array
        Length n vector with complex entries whose real and imaginary part is independently drawn
        from the normal distribution with mean 0 and variance 1.
    """
    # randn(n) produces a random vector of length n with mean 0 and variance 1
    g = np.random.randn(n) + 1j * np.random.randn(n)
    g = g / np.linalg.norm(g)
    return g


def randHerm(n):
    """Generate random square hermitian matrix

    Parameters
    ----------
    n : int
        Matrix dimension

    Returns
    -------
    G: 2D numpy array
        Random hermitian matrix normalized in spectral norm

    Notes:
        First a matrix with random complex entries is generated (see randvec function).
        This matrix is then projected onto the space of hermitian
        matrices and normalized in spectral norm.
    """
    G = randvec(n * n).reshape(n, n)
    G = (G + G.T.conj()) / 2
    # ord=2 gives the spectral norm
    G = G / np.linalg.norm(G, ord=2)
    return G


def randHermGS(d, r):
    """Generates random set of operators that are hermiticity preserving

    Parameters
    ----------
    d : int
        Number of gates
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension

    Returns
    -------
    X: 3D numpy array
        Array where random hermiticity preserving operators are stacked along the first axis.

    Notes:
        The function randHerm is used to generate random hermitian Choi matrices,
        whose indies are then rearranged to obtain random hermiticity preserving superoperators.
    """
    dim = int(np.sqrt(r))
    X = np.zeros((r, d, r), dtype="complex")
    for i in range(d):
        H = randHerm(r).reshape(dim, dim, dim, dim)
        H = np.einsum("ijkl->jlik", H)
        X[:, i, :] = H.reshape(r, r)
    return X


def randU(n, a=1):
    """Generates random unitary from a random hermitian generator

    Parameters
    ----------
    n : int
        Matrix dimension of the unitary
    a : float
        Parameter to control the norm of the hermitian generator

    Returns
    -------
    U: 2D numpy array
        Matrix exponential of random hermitian matrix times the imaginary unit.
    """
    return expm(1j * a * randHerm(n)).astype(np.complex128)


def randU_Haar(n):
    """Return a Haar distributed random unitary

    Parameters
    ----------
    n : int
        Matrix dimension of the unitary

    Returns
    -------
    U: 2D numpy array
        Random unitary matrix distributed according to the Haar measure.
    """
    Z = np.random.randn(n, n) + 1.0j * np.random.randn(n, n)
    [Q, R] = qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)


def randKrausSet(d, r, rK, a=1):
    """Generates random set of Kraus operators

    Parameters
    ----------
    d : int
        Number of gates
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Number of Kraus operators per gate ("Kraus rank")
    a : float
        Parameter to control the norm of the hermitian generator and thereby
        how far the gates are from the identity

    Returns
    -------
    K: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Notes:
        Let pdim be the physical dimension. Then a set of Kraus operators is generated
        by taking the first pdim columns of a random unitary of size rK*pdim.
        The random unitary is generated from a random hermitian matrix.
    """
    pdim = int(np.sqrt(r))
    K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    for i in range(d):
        K[i, :, :, :] += randU(pdim * rK, a)[:, :pdim].reshape(rK, pdim, pdim)
    return K


def randKrausSet_Haar(d, r, rK):
    """Generates random set of Kraus operators

    Parameters
    ----------
    d : int
        Number of gates
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Number of Kraus operators per gate ("Kraus rank")

    Returns
    -------
    K: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Notes:
        Let pdim be the physical dimension. Then a set of Kraus operators is
        generated by taking the first pdim columns of a random unitary of size
        rK*pdim. The random unitary is generated according the the Haar measure.
    """
    pdim = int(np.sqrt(r))
    K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    for i in range(d):
        K[i, :, :, :] += randU_Haar(pdim * rK)[:, :pdim].reshape(rK, pdim, pdim)
    return K


def random_gs(d, r, rK, n_povm):
    """Generates a random gate using the Gaussian unitary ensemble, initial state and POVM

    Parameters
    ----------
    d : int
        Number of gates
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Number of Kraus operators per gate ("Kraus rank")
    n_povm : int
        Number of POVM-Elements

    Returns
    -------
    K: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    X: 3D numpy array
        Array where random CPT superoperators are stacked along the first axis.

    Notes:
        The Kraus operators are generated from random unitaries, see function randKrausSet
    """
    K = randKrausSet(d, r, rK).copy()
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape(d, r, r)
    rho = randpsd(r).copy()
    A = randKrausSet(1, r, n_povm)[0].conj()
    E = np.array([(A[i].T.conj() @ A[i]).reshape(-1) for i in range(n_povm)]).copy()
    return K, X, E, rho


def random_gs_Haar(d, r, rK, n_povm):
    """Generates a random gate set with gates from Haar random unitaries, initial state and POVM

    Parameters
    ----------
    d : int
        Number of gates
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Number of Kraus operators per gate ("Kraus rank")
    n_povm : int
        Number of POVM-Elements

    Returns
    -------
    K: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    X: 3D numpy array
        Array where random CPT superoperators are stacked along the first axis.

    Notes:
        The Kraus operators are generated from Haar random unitaries, see function randKrausSet_Haar
    """
    K = randKrausSet_Haar(d, r, rK).copy()
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape(d, r, r)
    rho = randpsd(r).copy()
    A = randKrausSet_Haar(1, r, n_povm)[0].conj()
    E = np.array([(A[i].T.conj() @ A[i]).reshape(-1) for i in range(n_povm)]).copy()
    return K, X, E, rho

def perturbed_target_init(X_target, rK):
    """Generates a small random noise gate around the identity and applies it to the target gate
    The reason for using this gate as an initialization as opposed the the target gate itself, is
    that the non-dominant Kraus operators we start with are now not zero, but small random matrices.
    Observations show that with this start, the non-dominant Kraus operators converge faster.

    Parameters
    ----------
    X_target : 3D numpy array
        Current gate estimate
    rK : int
        Number of Kraus operators per gate ("Kraus rank") for the initialization

    Returns
    -------
    K_init: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    """
    d,r,_ = X_target.shape
    pdim = int(np.sqrt(r))
    K_perturb = randKrausSet(d, r, rK, a=0.1)
    X_perturb = np.einsum("ijkl,ijnm -> iknlm", K_perturb, K_perturb.conj()).reshape(d, r, r)
    X_init = np.einsum('ikl,ilm ->ikm', X_perturb, X_target)
    K_init = Kraus_rep(X_init, d, pdim, rK)
    return K_init
def basis(size, index):
    """Creates standard basis vectors

    Parameters
    ----------
    size : int
        Vector space dimension
    index : int
        Index of basis vector

    Returns
    -------
    vec: 1D numpy array
        Vector with entry 1 at position given by index and zeros elsewhere
    """
    vec = np.zeros(size)
    vec[index] = 1.0
    return vec


def depol(pdim, p):
    """Kraus representation of depolarizing channel

    Parameters
    ----------
    pdim : int
        Physical dimension
    p : float
        Error probability

    Returns
    -------
    K_depol: 4D numpy array
        Each entry along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Notes:
        The depolarizing channel is defined as L(rho) = (1-p)*rho + p/pdim*Id.
    """
    phi_plus = np.sum([np.kron(basis(pdim, i), basis(pdim, i)) for i in range(pdim)], axis=0)
    choi_state = p / pdim * np.eye(pdim**2) + (1 - p) * np.kron(
        phi_plus, phi_plus.reshape(pdim**2, 1))
    K_depol = la.cholesky(choi_state)
    return K_depol.reshape(pdim, pdim, pdim**2).swapaxes(0, 2)


def varassign(v, X, E, rho, argument):
    """Assigns input to specified gate set variables

    Parameters
    ----------
    v : numpy array
        New set of variables
    X : numpy array
        Current gate estimate
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    argument : {"X", "E", "rho"}
        Which part of the gate set is updated

    Returns
    -------
    [.,.,.]: 3 element list
        List in the order [X,E,rho] where either X, E or rho is repaced by v,
        depending on the input to the "arguement" variable
    """
    if argument == "X" or argument == "K":
        return [v, E, rho]
    elif argument == "E":
        return [X, v, rho]
    elif argument == "rho":
        return [X, E, v]


def batch(y, J, bsize):
    """Returns random batch of sequences and corresponding measurements

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    bsize : int
        Size of the batch (number of sequences)

    Returns
    -------
    y_b : numpy array
        Randomly subsampled columns of y
    J_b : numpy array
        Randomly subsampled rows of J in accordance the the columns selected in y_b
    """
    if y.shape[1] <= bsize:
        return y, J
    if bsize < 1:  # if batch size is given as ratio
        bsize = int(bsize * len(J) // 1)
    batchmask = np.array([1] * bsize + [0] * (len(J) - bsize))
    np.random.shuffle(batchmask)
    J_b = J[batchmask == 1]
    y_b = y[:, batchmask == 1]
    return y_b, J_b


def F_avg_X(X, K):
    """Returns the average gate fidelity between two gates given by a
    superoperator and a set of Kraus operators

    Parameters
    ----------
    X : 2D numpy array
        CPT superoperator of size (pysical dimension**2) x (pysical dimension **2)
    K : 3D numpy array
        Array of Kraus operators with size (Kraus rank) x (pysical dimension) x (pysical dimension)

    Returns
    -------
    Fid : float
        Average gate fidelity

    Notes:
    Not gauge optimization involved; Average gate fidelity is gauge dependent.
    """
    pdim = K.shape[2]
    d = K.shape[0]
    Fid_list = []
    for k in range(d):
        choi_inner_prod = np.einsum(
            "imjl,pml,pij", X[k].reshape(pdim, pdim, pdim, pdim), K[k], K[k].conj())

        unitality_term = np.einsum(
            "imkk,pml,pil", X[k].reshape(pdim, pdim, pdim, pdim), K[k], K[k].conj())
        Fid = (choi_inner_prod + unitality_term) / pdim / (pdim + 1)
        Fid_list.append(Fid)
    return np.average(np.real(Fid_list)), np.real(Fid_list)


def MVE(X_true, E_true, rho_true, X, E, rho, d, length, n_povm, samples=10000):
    """Mean varation error between the outputs of two gate sets on random sequences

    Parameters
    ----------
    X_true : numpy array
        Target gates
    E_true : numpy array
        Target POVM
    rho_true : numpy array
        Target initial state
    X : numpy array
        Current gate estimate
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    d : int
        Number of different gates in the gate set
    length : int
        Length of the test sequences
    n_povm : int
        Number of POVM elements
    samples : int
        Number of random gate sequences over which the mean variation error is computed

    Returns
    -------
    MVE : float
        Mean varaition error

    Notes:
        Sequences are drawn without replacement from initally d**l possibilities.
        For each sequence the total variation error of the two probability distribution
        over the POVM elements is computed. Afterwards the meean over these total
        variation errors is returned.
    """
    if samples == "all" or np.log(samples) / np.log(d) > length:
        J = np.random.randint(0, d, length * d**length).reshape(d**length, length)
    else:
        J = np.random.randint(0, d, length * samples).reshape(samples, length)
    return MVE_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm)


def Mp_norm(X_true, E_true, rho_true, X, E, rho, d, length, n_povm, p, samples=10000):
    """Mean of the p-norm deviation between the outputs of two gate sets on random sequences

    Parameters
    ----------
    X_true : numpy array
        Target gates
    E_true : numpy array
        Target POVM
    rho_true : numpy array
        Target initial state
    X : numpy array
        Current gate estimate
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    d : int
        Number of different gates in the gate set
    length : int
        Length of the test sequences
    n_povm : int
        Number of POVM elements
    p : int
        Defines the l_p - norm that is used to compare probability distributions
    samples : int
        Number of random gate sequences over which the mean variation error is computed

    Returns
    -------
    MPE : float
        Mean l_p - norm error

    Notes:
        Sequences are drawn without replacement from initally d**l possibilities.
        For each sequence the l_p - norm error of the two probability distribution
        over the POVM elements is computed. Afterwards the meean over these total
        variation errors is returned.

    """
    if samples == "all" or np.log(samples) / np.log(d) > length:
        J = np.random.randint(0, d, length * d**length).reshape(d**length, length)
    else:
        J = np.random.randint(0, d, length * samples).reshape(samples, length)
    return Mp_norm_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm, p)


def Kraus_rep(X, d, pdim, rK):
    """Compute the Kraus representations for all gates in the gate set

    Parameters
    ----------
    X : numpy array
        Current gate estimate
    d : int
        Number of gates
    pdim : int
        Physical dimension
    rK : int
        Target Kraus rank

    Returns
    -------
    K: 4D numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Notes:
        The Kraus representation is obtained from a singular value decomposition
        of the Choi matrix. If parameter rK is smaller than the true rank of the
        Choi matrix, a rank rK approximation is used. Approximations are only CPT
        in the special case where the original gate was already of rank <= rK.
    """
    X_choi = X.reshape(d, pdim, pdim, pdim, pdim)
    X_choi = np.einsum("ijklm->iljmk", X_choi).reshape(d, pdim**2, pdim**2)
    K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    for i in range(d):
        w, v = la.eigh(X_choi[i])
        if np.min(w) < -1e-12:
            raise ValueError("Choi Matrix is not positive definite within tolerance 1e-12")
        K[i] = np.einsum("ijk->kji",
                         (v[:, -rK:] @ np.diag(np.sqrt(np.abs(w[-rK:])))).reshape(pdim, pdim, rK))
        # Trace normalization of Choi Matrix
        K[i] = (K[i] / np.sqrt(np.einsum("ijk,ijk", K[i], K[i].conj())) * np.sqrt(pdim))
    return np.array(K)


def sampled_measurements(y, n):
    """Compute finite sample estimates of input probabilities

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for different sequences;
        Each column contains the outcome probabilities for a fixed sequence
    n : Number of samples for each experiment

    Returns
    -------
    y_sampled : numpy array
        2D array of sampled measurement outcomes for different sequences;
        Each column contains the sampled outcome probabilities for a fixed sequence

    Notes:
        The entries of each column of y form a probability distribution.
        From this distribution n random samples are drawn which give estimates for
        the initial probabilities. This simulates finite sample size data.
    """
    n_povm, m = y.shape
    if any(y.reshape(-1) > 1) or any(y.reshape(-1) < 0):
        y_new = np.maximum(np.minimum(y, 1), 0)
        if np.sum(np.abs(y_new - y)) > 1e-6:
            warnings.warn("Warning: Probabilities capped to interval [0,1]",
                          "l1-difference to input:%f" % np.sum(np.abs(y_new - y)))
        y = y_new
    rng = np.random.default_rng()
    y_sampled = np.zeros(y.shape)
    for i in range(m):
        y_sampled[:, i] = rng.multinomial(n, [y[o, i] for o in range(n_povm)]) / n
    return y_sampled


def random_len_seq(d, min_l, max_l, N):
    """Generate random gate sequence instructions which contain sequences of different lengths,
    the lengths are drawn uniformly at random from (min_l, ..., max_l)

    Parameters
    ----------
    d : int
        Number of gates
    min_l : int
        Minimum sequence length
    max_l : int
        Maximum sequence length
    N : int
        Number of random sequences

    Returns
    -------
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence

    """
    seq_lengths = np.random.randint(min_l, max_l + 1, N)
    J = []
    for length in seq_lengths:
        j_curr = np.random.randint(0, d, length)
        J.append(list(np.pad(j_curr, (0, max_l - length), "constant", constant_values=-1)))
    return np.array(J)


def generate_fids(d, length, m_f):
    """Generate random fiducial sequencecs

    Parameters
    ----------
    d : int
        Number of gates
    length : int
        Total sequence length
    m_f : int
        Number of random fiducial sequences

    Returns
    -------
    J_fid : numpy array
        Sequence list of only the fiducial sequences
    J_fid2 : numpy array
        Sequence list for all combinations of two concatenated fiducial sequences
    J_meas : numpy array
        Sequence list for all combinations of fiducials seuqneces with a gate
        in between: fiducial1 -- gate -- fiducial2
    """
    fid_length = (length - 1) // 2
    fid = random.sample(range(d**fid_length), m_f).copy()
    J_fid = [list(local_basis(ind, d, fid_length)) for ind in fid]
    J_fid2 = np.array([seqL + seqR for seqL in J_fid for seqR in J_fid])
    J_meas = np.zeros((d, m_f**2, length), dtype="int")
    for k in range(d):
        J_meas[k] = np.array([seqL + [k] + seqR for seqL in J_fid for seqR in J_fid])
    return np.array(J_fid), J_fid2, J_meas


def is_positive(X, E, rho):
    """Print the results for checks whether a gate set is physical.

    This includes all positivity and normalization constraints.

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    """
    d, r, _ = X.shape
    pdim = int(np.sqrt(r))
    n_povm = E.shape[0]

    X_choi = X.reshape(d, pdim, pdim, pdim, pdim)
    X_choi = np.einsum("ijklm->iljmk", X_choi).reshape(d, r, r)

    eigvals = np.array([la.eigvals(X_choi[i]) for i in range(d)])
    partial_traces = np.einsum("aiikl -> akl", X.reshape(d, pdim, pdim, pdim, pdim))
    povm_eigvals = np.array([la.eigvals(E[i].reshape(pdim, pdim)) for i in range(n_povm)])
    if np.any(np.imag(eigvals.reshape(-1) > 1e-10)):
        print("Gates are not all hermitian.")
    else:
        for i in range(d):
            print("Gate %i positive:" % i, np.all(eigvals[i, :] > -1e-10))
            print("Gate %i trace preserving:" % i,
                  la.norm(partial_traces[i] - np.eye(pdim)) < 1e-10)
    print("Initial state positive:", np.all(la.eigvals(rho.reshape(pdim, pdim)) > -1e-10))
    print("Initial state normalization:", np.trace(rho.reshape(pdim, pdim)))
    print("POVM valid:", np.all([la.norm(np.sum(E, axis=0).reshape(pdim, pdim)
                                         - np.eye(pdim)) < 1e-10,
                                 np.all(povm_eigvals.reshape(-1) > -1e-10)]))
    return


def tvd(X, E, rho, J, y_data):
    """Return the total variation distance between model probabilities for the circuits in J
    and the probabilities given by y_data.

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    y_data : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    bsize : int
        Size of the batch (number of sequences)

    Returns
    -------
    dist : float
        The total variation distance.
    """
    n_povm = y_data.shape[0]
    y_model = np.real(np.array([[E[i].conj() @ contract(X, j) @ rho for j in J]
                                for i in range(n_povm)]))
    dist = la.norm(y_model - y_data, ord=1) / 2
    return dist


def random_seq_design(d, l_min, l_cut, l_max, N_short, N_long): #Draws without replacement but ineficiently (not working for sequence length > 24)
    """ Generate a set of random sequences ith given lengths
    This sequence lengths (circuit depths) are chosen for a mix of very short sequences (better convergence) and some
    slightly longer sequences to reduce the generalization error.
    This sequence design is heuristic and intended for coarse and fast estimates. For very accurate estimates at the
    cost of higher measurement effort it is recommended to use pyGSTi with long sequence GST.

    Parameters
    ----------
    d : int
        The number of gates in the gate set
    l_min : int
        Minimum sequence lenght
    l_cut : int
        Cutoff sequence lenght: N_short sequences are equally distributed among lengths l_min < l < l_cut
    l_max : int
        N_long sequences are equally distributed among lengths l_cut + 1 < l < l_max. Currently l_max < 24 only.
    N_short : int
        Number of short sequences
    N_long : int
        Number of long sequences

    Returns
    -------
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence

    """
    # Open problems:
    # - Change randomness to work with longer sequences;
    # - Handle case where Number of sequences is smaller than the available lengths
    if l_max >= 24:
        raise ValueError("Currently only sequences lenghts < 24 are supported.")

    J = [-np.ones(l_max)]
    #Short sequences:
    seq_counts = []
    N_remaining = N_short
    for l in range(l_min, l_cut+1):
        seq_counts.append(int(np.min([np.floor(N_remaining/(l_cut+1-l)), d**l])))
        ind_curr = np.array(random.sample(range(d**l), seq_counts[-1]))
        J_curr = np.array([np.pad(local_basis(ind,d,l),(0,l_max-l),'constant', constant_values = -1) for ind in ind_curr])
        J = np.concatenate((J,J_curr), axis = 0)
        N_remaining = N_short - 1 - np.sum(seq_counts)
    if N_remaining > 0:
        print('Number of possible sequences without replacement for the given sequence\
        length range is lower than the desired total number of sequences')

    #Long sequences:
    seq_counts = []
    N_remaining = N_long
    for l in range(l_cut+1, l_max+1):
        seq_counts.append(int(np.min([np.floor(N_remaining/(l_max+1-l)), d**l])))
        ind_curr = np.array(random.sample(range(d**l), seq_counts[-1]))
        J_curr = np.array([np.pad(local_basis(ind,d,l),(0,l_max-l),'constant', constant_values = -1) for ind in ind_curr])
        J = np.concatenate((J,J_curr), axis = 0)
        N_remaining = N_long - np.sum(seq_counts)
    return J.astype(int)
