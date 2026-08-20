#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import csr_matrix, kron, identity

# ============================================================
# Single-site field operators from ladder operators
# ============================================================

def single_site_field_operators(nq, mu=1.0):

    # --------------------------------------------------------
    # Local Hilbert dimension
    # --------------------------------------------------------

    Nphi = 2**nq   # 64 for nq=6

    # --------------------------------------------------------
    # Ladder operator a
    # --------------------------------------------------------

    a = np.zeros((Nphi, Nphi), dtype=complex)

    for n in range(1, Nphi):

        a[n-1, n] = np.sqrt(n)

    # creation operator
    adag = a.conj().T

    # --------------------------------------------------------
    # Field operator
    #
    # Phi = (a + adag)/sqrt(2 mu)
    # --------------------------------------------------------

    Phi = (
        a + adag
    ) / np.sqrt(2 * mu)

    # --------------------------------------------------------
    # Conjugate momentum operator
    #
    # Pi = -i sqrt(mu/2) (a - adag)
    # --------------------------------------------------------

    Pi = (
        -1j
        * np.sqrt(mu / 2)
        * (a - adag)
    )

    # --------------------------------------------------------
    # Numerical Hermiticity cleanup
    # --------------------------------------------------------

    Phi = 0.5 * (Phi + Phi.conj().T)

    Pi = 0.5 * (Pi + Pi.conj().T)

    return csr_matrix(Phi), csr_matrix(Pi)


# ============================================================
# Two-site embedded operators
# ============================================================

def lattice_operators(n_sites, nq, mu=1.0):

    # Single-site operators
    Phi_local, Pi_local = single_site_field_operators(nq, mu)

    Nphi = 2**nq

    I = identity(
        Nphi,
        format='csr',
        dtype=complex
    )

    Phi = []
    Pi = []

    # Embed operators on each lattice site
    for site in range(n_sites):

        Phi_site = None
        Pi_site = None

        for j in range(n_sites):

            if j == site:
                Aphi = Phi_local
                Api = Pi_local
            else:
                Aphi = I
                Api = I

            if Phi_site is None:
                Phi_site = Aphi
                Pi_site = Api
            else:
                Phi_site = kron(Phi_site, Aphi, format='csr')
                Pi_site = kron(Pi_site, Api, format='csr')

        Phi.append(Phi_site)
        Pi.append(Pi_site)

    return Phi, Pi

n_sites = 6
nq = 4
mu = 1.0

Phi, Pi = lattice_operators(
    n_sites,
    nq,
    mu
)

print("Hilbert dimension =", Phi[0].shape[0])

for i in range(n_sites):
    print(f"Phi[{i}] shape =", Phi[i].shape)
    print(f"Pi[{i}] shape  =", Pi[i].shape)


# In[2]:


import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh

# ============================================================
# Local single-site Hamiltonian
# ============================================================

def single_site_hloc(Phi, Pi, mI2, lam, f):

    Phi2 = Phi @ Phi
    Phi4 = Phi2 @ Phi2
    Pi2  = Pi @ Pi

    hloc = (
        0.5 * Pi2
        + 0.5 * mI2 * Phi2
        + (lam / 24.0) * Phi4
        + f * Phi
    )

    return hloc




# In[3]:


# ============================================================
# Lattice local Hamiltonian
# ============================================================

def lattice_hloc(n_sites,
                 nq,
                 mu=1.0,
                 mI2=1.0,
                 lam=1.0,
                 f=1e-3):

    # --------------------------------------------------------
    # Embedded field operators
    # --------------------------------------------------------

    Phi, Pi = lattice_operators(
        n_sites,
        nq,
        mu
    )

    # --------------------------------------------------------
    # Sum of local Hamiltonians
    # --------------------------------------------------------

    Hloc = 0

    for i in range(n_sites):

        Hloc += single_site_hloc(
            Phi[i],
            Pi[i],
            mI2,
            lam,
            f
        )

    return Hloc


# In[4]:


def ground_state(H, tol=1e-10, maxiter=None):

    evals, evecs = eigsh(
        H,
        k=1,
        which='SA',
        tol=tol,
        maxiter=maxiter
    )

    E0 = np.real(evals[0])
    psi0 = evecs[:, 0]

    psi0 = psi0 / np.linalg.norm(psi0)

    return E0, psi0


# In[ ]:


# ============================================================
# Example
# ============================================================

n_sites = 6
nq      = 4
mu      = 1.0
mI2     = 1.0
lam     = 1.0
f       = 0.00005

Hloc = lattice_hloc(
    n_sites=n_sites,
    nq=nq,
    mu=mu,
    mI2=mI2,
    lam=lam,
    f=f
)

E0_loc, psi_loc = ground_state(Hloc)

print("Number of sites     =", n_sites)
print("Qubits per site     =", nq)
print("Total qubits        =", n_sites * nq)
print("Hilbert dimension   =", 2**(n_sites * nq))
print("Ground-state energy =", E0_loc)
print("State dimension     =", psi_loc.shape)
print("Norm                =", np.linalg.norm(psi_loc))


# In[5]:


# ============================================================
# Convert scalar-field Hamiltonian to Pauli form
# ============================================================

import numpy as np

from qiskit.quantum_info import (
    Operator,
    SparsePauliOp
)


# ============================================================
# Parameters
# ============================================================

n_sites = 6
nq      = 4

mu  = 1.0
mI2 = 1.0
lam = 1.0
f   = 1e-5


# ============================================================
# Build SINGLE-SITE local Hamiltonian
#
# This is only 16 x 16 because nq = 4
# ============================================================

Phi_local, Pi_local = single_site_field_operators(
    nq=nq,
    mu=mu
)

hloc = single_site_hloc(
    Phi_local,
    Pi_local,
    mI2=mI2,
    lam=lam,
    f=f
)

print("Single-site Hamiltonian shape =", hloc.shape)


# ============================================================
# Convert SINGLE-SITE Hamiltonian to Pauli form
#
# This is safe because the local Hilbert dimension is only
# 2^4 = 16.
# ============================================================

print("\nConverting single-site Hamiltonian to Pauli form...\n")

hloc_dense = hloc.toarray()

hloc_op = Operator(hloc_dense)

hloc_pauli = SparsePauliOp.from_operator(
    hloc_op
)

# Remove numerically negligible coefficients
hloc_pauli = hloc_pauli.simplify(
    atol=1e-10
)

print("Single-site Pauli terms =",
      len(hloc_pauli))


# ============================================================
# Embed the single-site Pauli Hamiltonian
# on all lattice sites
# ============================================================

global_pauli_terms = []

identity_block = "I" * nq

for site in range(n_sites):

    for pauli, coeff in zip(
        hloc_pauli.paulis,
        hloc_pauli.coeffs
    ):

        local_label = pauli.to_label()

        # ----------------------------------------------------
        # Put the local Pauli string on the chosen site
        # and identities on all other sites.
        # ----------------------------------------------------

        left_identity  = "I" * (site * nq)
        right_identity = "I" * ((n_sites - site - 1) * nq)

        global_label = (
            left_identity
            + local_label
            + right_identity
        )

        global_pauli_terms.append(
            (global_label, coeff)
        )


# ============================================================
# Construct full 24-qubit SparsePauliOp
# ============================================================

Hloc_pauli = SparsePauliOp.from_list(
    global_pauli_terms
)


# ============================================================
# Combine duplicate Pauli strings and remove tiny terms
# ============================================================

Hloc_pauli = Hloc_pauli.simplify(
    atol=1e-10
)


# ============================================================
# Information
# ============================================================

print("\nDone.")

print("\nNumber of sites       =", n_sites)
print("Qubits per site       =", nq)
print("Total number of qubits =", n_sites * nq)

print(
    "Number of Pauli terms =",
    len(Hloc_pauli)
)


# ============================================================
# Show first few Pauli terms
# ============================================================

print("\nFirst few Pauli terms:\n")

for pauli, coeff in zip(
    Hloc_pauli.paulis[:10],
    Hloc_pauli.coeffs[:10]
):

    print(f"{coeff.real:.12e} * {pauli}")


# In[6]:


# ============================================================
# Remove identity offset
# ============================================================

coeffs = Hloc_pauli.coeffs
paulis = Hloc_pauli.paulis

new_coeffs = []
new_paulis = []

identity_shift = 0.0

# Total number of qubits
nqubits = n_sites * nq

# Full-system identity
identity_string = 'I' * nqubits

for p, c in zip(paulis, coeffs):

    if p.to_label() == identity_string:

        identity_shift += np.real(c)

    else:

        new_paulis.append(p)
        new_coeffs.append(c)


# ============================================================
# Construct shifted Hamiltonian
# ============================================================

Hloc_pauli_shifted = SparsePauliOp(
    new_paulis,
    coeffs=new_coeffs
)


# ============================================================
# Remove tiny coefficients
# ============================================================

Hloc_pauli_shifted = Hloc_pauli_shifted.simplify(
    atol=1e-10
)


# ============================================================
# Print information
# ============================================================

print("Total qubits              =", nqubits)

print("Removed identity shift    =",
      identity_shift)

print("Remaining Pauli terms     =",
      len(Hloc_pauli_shifted))


# In[7]:


# ============================================================
# Hamiltonian Variational Ansatz (Improved Asymmetric HVA)
# ============================================================

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector 
from qiskit_algorithms import VQE 
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
# ============================================================
# Physics-inspired HVA for mI² < 0
# ============================================================

def grouped_hva_asymm(nq, layers):

    qc = QuantumCircuit(nq)

    # --------------------------------------------------------
    # Symmetry-breaking seed
    # --------------------------------------------------------

    for q in range(nq):

        qc.ry(0.01, q)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    nparams_per_layer = 3*nq + 3

    params = ParameterVector(
        "θ",
        nparams_per_layer * layers
    )

    for l in range(layers):

        offset = l * nparams_per_layer

        θx  = params[offset : offset+nq]

        θy  = params[offset+nq : offset+2*nq]

        θz  = params[offset+2*nq : offset+3*nq]

        θxx = params[offset+3*nq]

        θyy = params[offset+3*nq+1]

        θzz = params[offset+3*nq+2]

        # ----------------------------------------------------
        # RX sector
        # ----------------------------------------------------

        for q in range(nq):

            qc.rx(2*θx[q], q)

        # ----------------------------------------------------
        # RY sector
        # ----------------------------------------------------

        for q in range(nq):

            qc.ry(2*θy[q], q)

        # ----------------------------------------------------
        # RZ sector
        # ----------------------------------------------------

        for q in range(nq):

            qc.rz(2*θz[q], q)

        # ----------------------------------------------------
        # XX sector
        # ----------------------------------------------------

        for q in range(0, nq-1, 2):

            qc.rxx(2*θxx, q, q+1)

        for q in range(1, nq-1, 2):

            qc.rxx(2*θxx, q, q+1)

        # ----------------------------------------------------
        # YY sector
        # ----------------------------------------------------

        for q in range(0, nq-1, 2):

            qc.ryy(2*θyy, q, q+1)

        for q in range(1, nq-1, 2):

            qc.ryy(2*θyy, q, q+1)

        # ----------------------------------------------------
        # ZZ sector
        # ----------------------------------------------------

        for q in range(0, nq-1, 2):

            qc.rzz(2*θzz, q, q+1)

        for q in range(1, nq-1, 2):

            qc.rzz(2*θzz, q, q+1)

    return qc


# In[ ]:


# ============================================================
# Hamiltonian Variational Ansatz (Improved HVA)
# ============================================================

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ============================================================
# Physics-inspired HVA
# ============================================================

def grouped_hva_symm(nq, layers):

    qc = QuantumCircuit(nq)

    # --------------------------------------------------------
    # Tiny symmetry-breaking seed
    # --------------------------------------------------------

    for q in range(nq):
        qc.ry(0.01, q)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    # RX, RY, RZ on every qubit
    # one shared XX, YY, ZZ angle

    nparams_per_layer = 3*nq + 3

    params = ParameterVector(
        "θ",
        nparams_per_layer * layers
    )

    for l in range(layers):

        offset = l * nparams_per_layer

        θx  = params[offset : offset+nq]

        θy  = params[offset+nq : offset+2*nq]

        θz  = params[offset+2*nq : offset+3*nq]

        θxx = params[offset+3*nq]

        θyy = params[offset+3*nq+1]

        θzz = params[offset+3*nq+2]

        # ====================================================
        # X rotations
        # ====================================================

        for q in range(nq):

            qc.rx(2*θx[q], q)

        # ====================================================
        # Y rotations
        # ====================================================

        for q in range(nq):

            qc.ry(2*θy[q], q)

        # ====================================================
        # Z rotations
        # ====================================================

        for q in range(nq):

            qc.rz(2*θz[q], q)

        # ====================================================
        # XX block
        # ====================================================

        for q in range(0, nq-1, 2):

            qc.rxx(2*θxx, q, q+1)

        for q in range(1, nq-1, 2):

            qc.rxx(2*θxx, q, q+1)

        # ====================================================
        # YY block
        # ====================================================

        for q in range(0, nq-1, 2):

            qc.ryy(2*θyy, q, q+1)

        for q in range(1, nq-1, 2):

            qc.ryy(2*θyy, q, q+1)

        # ====================================================
        # ZZ block
        # ====================================================

        for q in range(0, nq-1, 2):

            qc.rzz(2*θzz, q, q+1)

        for q in range(1, nq-1, 2):

            qc.rzz(2*θzz, q, q+1)

    return qc


# In[8]:


###### Exact Ground State Utility

# ============================================================
# Exact ground state from sparse Hamiltonian
# ============================================================

from scipy.sparse.linalg import eigsh


def exact_ground_state(H_sparse, tol=1e-10, maxiter=None):

    evals, evecs = eigsh(
        H_sparse,
        k=1,
        which='SA',
        tol=tol,
        maxiter=maxiter
    )

    E0 = np.real(evals[0])

    psi0 = evecs[:, 0]
    psi0 = psi0 / np.linalg.norm(psi0)

    return E0, psi0


# In[10]:


# ============================================================
# Fidelity
# ============================================================

def fidelity(psi_exact, psi_vqe):

    if psi_exact.shape != psi_vqe.shape:
        raise ValueError(
            f"State dimension mismatch: "
            f"exact = {psi_exact.shape}, "
            f"VQE = {psi_vqe.shape}"
        )

    psi_exact = psi_exact / np.linalg.norm(psi_exact)
    psi_vqe   = psi_vqe / np.linalg.norm(psi_vqe)

    return np.abs(
        np.vdot(psi_exact, psi_vqe)
    )**2


# In[11]:


import numpy as np
from scipy.sparse import identity, kron

# ============================================================
# N-site Hamiltonian decomposition
# ============================================================

def lattice_hloc_decomposed(
    n_sites,
    nq,
    mu=1.0,
    mI2=1.0,
    lam=1.0,
    f=1e-3
):
    """
    Returns the individual local Hamiltonian pieces

        Hpi_sparse
        Hphi2_sparse
        Hphi4_sparse
        Hf_sparse
        Htotal_sparse

    for an n_sites lattice.

    The local Hamiltonian is

        Hloc = Hpi + Hphi2 + Hphi4 + Hf

    with

        Hpi   = sum_i 1/2 Pi_i^2
        Hphi2 = sum_i 1/2 mI2 Phi_i^2
        Hphi4 = sum_i (lambda/24) Phi_i^4
        Hf    = sum_i f Phi_i
    """

    # --------------------------------------------------------
    # Single-site operators
    # --------------------------------------------------------

    Phi, Pi = single_site_field_operators(
        nq,
        mu
    )

    # --------------------------------------------------------
    # Powers of field operators
    # --------------------------------------------------------

    Phi2 = Phi @ Phi
    Phi4 = Phi2 @ Phi2
    Pi2  = Pi @ Pi

    # --------------------------------------------------------
    # Single-site Hilbert dimension
    # --------------------------------------------------------

    N = 2**nq

    I = identity(
        N,
        format="csr",
        dtype=complex
    )

    # --------------------------------------------------------
    # Single-site Hamiltonian pieces
    # --------------------------------------------------------

    h_pi = 0.5 * Pi2

    h_phi2 = 0.5 * mI2 * Phi2

    h_phi4 = (lam / 24.0) * Phi4

    h_f = f * Phi

    # --------------------------------------------------------
    # Initialize full lattice Hamiltonians
    # --------------------------------------------------------

    Hpi_sparse = None
    Hphi2_sparse = None
    Hphi4_sparse = None
    Hf_sparse = None

    # --------------------------------------------------------
    # Embed each single-site piece
    #
    # H = sum_i h_i
    # --------------------------------------------------------

    for site in range(n_sites):

        # ----------------------------------------------------
        # Construct identity/operator factors
        # ----------------------------------------------------

        hpi_site   = None
        hphi2_site = None
        hphi4_site = None
        hf_site    = None

        for j in range(n_sites):

            if j == site:

                A_pi   = h_pi
                A_phi2 = h_phi2
                A_phi4 = h_phi4
                A_f    = h_f

            else:

                A_pi   = I
                A_phi2 = I
                A_phi4 = I
                A_f    = I

            # ------------------------------------------------
            # Kronecker products
            # ------------------------------------------------

            if hpi_site is None:

                hpi_site   = A_pi
                hphi2_site = A_phi2
                hphi4_site = A_phi4
                hf_site    = A_f

            else:

                hpi_site = kron(
                    hpi_site,
                    A_pi,
                    format="csr"
                )

                hphi2_site = kron(
                    hphi2_site,
                    A_phi2,
                    format="csr"
                )

                hphi4_site = kron(
                    hphi4_site,
                    A_phi4,
                    format="csr"
                )

                hf_site = kron(
                    hf_site,
                    A_f,
                    format="csr"
                )

        # ----------------------------------------------------
        # Add contribution from this site
        # ----------------------------------------------------

        if Hpi_sparse is None:

            Hpi_sparse = hpi_site
            Hphi2_sparse = hphi2_site
            Hphi4_sparse = hphi4_site
            Hf_sparse = hf_site

        else:

            Hpi_sparse += hpi_site
            Hphi2_sparse += hphi2_site
            Hphi4_sparse += hphi4_site
            Hf_sparse += hf_site

    # --------------------------------------------------------
    # Total local Hamiltonian
    # --------------------------------------------------------

    Htotal_sparse = (
        Hpi_sparse
        + Hphi2_sparse
        + Hphi4_sparse
        + Hf_sparse
    )

    return (
        Hpi_sparse,
        Hphi2_sparse,
        Hphi4_sparse,
        Hf_sparse,
        Htotal_sparse
    )


# In[13]:


import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ============================================================
# Remove identity term from a SparsePauliOp
# ============================================================

def remove_identity(pauli_op):

    coeffs = []
    paulis = []

    identity_shift = 0.0

    # --------------------------------------------------------
    # Number of qubits in the Pauli operator
    # --------------------------------------------------------

    nqubits = pauli_op.num_qubits

    identity_string = "I" * nqubits

    # --------------------------------------------------------
    # Separate identity term from non-identity terms
    # --------------------------------------------------------

    for p, c in zip(
        pauli_op.paulis,
        pauli_op.coeffs
    ):

        if p.to_label() == identity_string:

            identity_shift += np.real(c)

        else:

            paulis.append(p)
            coeffs.append(c)

    # --------------------------------------------------------
    # Construct shifted operator
    # --------------------------------------------------------

    if len(paulis) == 0:

        shifted = SparsePauliOp(
            [identity_string],
            coeffs=[0.0]
        )

    else:

        shifted = SparsePauliOp(
            paulis,
            coeffs=coeffs
        )

    return shifted, identity_shift


# In[ ]:


# ============================================================
# Embed a local Pauli operator on all lattice sites
# ============================================================

from qiskit.quantum_info import SparsePauliOp


def embed_local_pauli_on_lattice(
    local_pauli,
    n_sites,
    nq
):
    """
    Embed a single-site Pauli Hamiltonian on every site.

    local_pauli:
        SparsePauliOp acting on nq qubits

    Returns:
        SparsePauliOp acting on n_sites*nq qubits
    """

    global_terms = []

    for site in range(n_sites):

        for pauli, coeff in zip(
            local_pauli.paulis,
            local_pauli.coeffs
        ):

            local_label = pauli.to_label()

            left_identity = "I" * (site * nq)

            right_identity = "I" * (
                (n_sites - site - 1) * nq
            )

            global_label = (
                left_identity
                + local_label
                + right_identity
            )

            global_terms.append(
                (global_label, coeff)
            )

    return SparsePauliOp.from_list(
        global_terms
    ).simplify(
        atol=1e-10
    )

# ===========================================================

# ============================================================
# Multi-parameter scan
# 6-site HVA + warm-start continuation + overlap tracking
# ============================================================

import numpy as np

from qiskit.quantum_info import (
    Operator,
    SparsePauliOp,
    Statevector
)

from qiskit_algorithms import VQE

from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.optimizers import L_BFGS_B


# ============================================================
# Fixed parameters
# ============================================================

n_sites = 6
nq_site = 4

# Total number of qubits
nqubits = n_sites * nq_site

mu = 1.0

# HVA depth
layers = 4


# ============================================================
# Number of VQE restarts
# ============================================================

n_restarts = 5


# ============================================================
# Energy tolerance
# ============================================================

energy_tol = 1e-3


# ============================================================
# Parameter sets
# ============================================================

parameter_sets = [

    {
        "label": "Symmetric",
        "mI2": 1.0,
        "f": 0.0
    },

    {
        "label": "Weak SSB optimised",
        "mI2": -1.0,
        "f": 5e-2
    },

    {
        "label": "Weak SSB",
        "mI2": -1.0,
        "f": 1e-4
    },

    {
        "label": "Strong SSB",
        "mI2": -1.0,
        "f": 0.5
    }
]


# ============================================================
# Lambda values
# ============================================================

lambda_list = np.linspace(
    0.1,
    2.0,
    10
)


# ============================================================
# Storage
# ============================================================

all_results = []


# ============================================================
# System information
# ============================================================

print("=" * 80)

print("SYSTEM INFORMATION")

print("=" * 80)

print(
    "Number of sites       =",
    n_sites
)

print(
    "Qubits per site       =",
    nq_site
)

print(
    "Total qubits          =",
    nqubits
)

print(
    "Hilbert dimension     =",
    2**nqubits
)

print(
    "HVA layers            =",
    layers
)

print("=" * 80)


# ============================================================
# Main parameter-set loop
# ============================================================

for params in parameter_sets:

    label = params["label"]

    mI2 = params["mI2"]

    f = params["f"]


    print("\n")
    print("#" * 80)

    print(
        f"CASE: {label}"
    )

    print(
        f"mI² = {mI2},  f = {f}"
    )

    print("#" * 80)


    case_results = []


    # ========================================================
    # Tracking variables
    # ========================================================

    previous_state = None

    previous_optimal_point = None


    # ========================================================
    # Lambda loop
    # ========================================================

    for lam in lambda_list:

        print("\n" + "=" * 60)

        print(
            f"Running λ = {lam:.3f}"
        )

        print("=" * 60)


        # ====================================================
        # Build 6-site H_loc
        # ====================================================

        (
            Hpi_sparse,
            Hphi2_sparse,
            Hphi4_sparse,
            Hf_sparse,
            Hloc_sparse

        ) = lattice_hloc_decomposed(

            n_sites=n_sites,

            nq=nq_site,

            mu=mu,

            mI2=mI2,

            lam=lam,

            f=f

        )


        # ====================================================
        # Exact ground state of H_loc
        # ====================================================

        E_exact, psi_exact = exact_ground_state(

            Hloc_sparse

        )


        # ====================================================
        # Construct LOCAL 16 x 16 Hamiltonian pieces
        #
        # We do NOT convert the 24-qubit Hloc to dense form.
        # ====================================================

        Phi_local, Pi_local = (
            single_site_field_operators(
                nq_site,
                mu
            )
        )


        Phi2_local = (
            Phi_local @ Phi_local
        )

        Phi4_local = (
            Phi2_local @ Phi2_local
        )

        Pi2_local = (
            Pi_local @ Pi_local
        )


        # ====================================================
        # Single-site Hamiltonian pieces
        # ====================================================

        hpi_local = (
            0.5 * Pi2_local
        )

        hphi2_local = (
            0.5 * mI2 * Phi2_local
        )

        hphi4_local = (
            (lam / 24.0) * Phi4_local
        )

        hf_local = (
            f * Phi_local
        )


        # ====================================================
        # Convert LOCAL pieces to Pauli form
        # ====================================================

        Hpi_local_pauli = (
            SparsePauliOp.from_operator(

                Operator(
                    hpi_local.toarray()
                )

            ).simplify(
                atol=1e-10
            )
        )


        Hphi2_local_pauli = (
            SparsePauliOp.from_operator(

                Operator(
                    hphi2_local.toarray()
                )

            ).simplify(
                atol=1e-10
            )
        )


        Hphi4_local_pauli = (
            SparsePauliOp.from_operator(

                Operator(
                    hphi4_local.toarray()
                )

            ).simplify(
                atol=1e-10
            )
        )


        Hf_local_pauli = (
            SparsePauliOp.from_operator(

                Operator(
                    hf_local.toarray()
                )

            ).simplify(
                atol=1e-10
            )
        )


        # ====================================================
        # Embed local Pauli pieces onto all 6 sites
        # ====================================================

        Hpi_pauli = (
            embed_local_pauli_on_lattice(
                Hpi_local_pauli,
                n_sites,
                nq_site
            )
        )


        Hphi2_pauli = (
            embed_local_pauli_on_lattice(
                Hphi2_local_pauli,
                n_sites,
                nq_site
            )
        )


        Hphi4_pauli = (
            embed_local_pauli_on_lattice(
                Hphi4_local_pauli,
                n_sites,
                nq_site
            )
        )


        Hf_pauli = (
            embed_local_pauli_on_lattice(
                Hf_local_pauli,
                n_sites,
                nq_site
            )
        )


        # ====================================================
        # Total 24-qubit H_loc in Pauli form
        # ====================================================

        H_pauli = (

            Hpi_pauli
            + Hphi2_pauli
            + Hphi4_pauli
            + Hf_pauli

        ).simplify(
            atol=1e-10
        )


        # ====================================================
        # Remove identity terms
        # ====================================================

        Hpi_pauli, _ = remove_identity(
            Hpi_pauli
        )

        Hphi2_pauli, _ = remove_identity(
            Hphi2_pauli
        )

        Hphi4_pauli, _ = remove_identity(
            Hphi4_pauli
        )

        Hf_pauli, _ = remove_identity(
            Hf_pauli
        )


        H_pauli_shifted, identity_shift = (
            remove_identity(
                H_pauli
            )
        )


        # ====================================================
        # HVA ANSATZ
        #
        # THIS IS THE IMPORTANT CHANGE
        #
        # Same HVA structure as the 2-site code.
        # Only the number of qubits has changed:
        #
        # 2 sites  -> 8 qubits
        # 6 sites  -> 24 qubits
        # ====================================================

        if mI2 < 0:

            ansatz = grouped_hva_asymm(

                nq=nqubits,

                layers=layers

            )

        else:

            ansatz = grouped_hva_symm(

                nq=nqubits,

                layers=layers

            )


        print(
            "HVA parameters =",
            ansatz.num_parameters
        )


        # ====================================================
        # Estimator
        # ====================================================

        estimator = StatevectorEstimator()


        # ====================================================
        # Optimizer
        # ====================================================

        optimizer = L_BFGS_B(

            maxiter=1000

        )


        # ====================================================
        # Weak SSB:
        # warm-start + overlap tracking
        # ====================================================

        if mI2 < 0:

            print(
                "\nRunning HVA-VQE "
                "with warm-start tracking...\n"
            )


            candidate_states = []


            # =================================================
            # Multiple VQE restarts
            # =================================================

            for trial in range(
                n_restarts
            ):

                print(
                    f"\nRestart {trial}"
                )


                # ---------------------------------------------
                # Initial point
                # ---------------------------------------------

                if (

                    trial == 0

                    and previous_optimal_point
                    is not None

                ):

                    initial_point = (
                        previous_optimal_point.copy()
                    )

                else:

                    # -----------------------------------------
                    # IMPORTANT:
                    # Use ansatz.num_parameters.
                    #
                    # Do NOT assume 3*layers parameters.
                    # The grouped HVA determines its own
                    # parameter count.
                    # -----------------------------------------

                    initial_point = (
                        0.01
                        * np.random.randn(
                            ansatz.num_parameters
                        )
                    )


                # ---------------------------------------------
                # VQE
                # ---------------------------------------------

                vqe = VQE(

                    estimator=estimator,

                    ansatz=ansatz,

                    optimizer=optimizer,

                    initial_point=initial_point

                )


                result = (
                    vqe.compute_minimum_eigenvalue(

                        H_pauli_shifted

                    )
                )


                # ---------------------------------------------
                # Physical energy
                # ---------------------------------------------

                E_trial = (

                    result.eigenvalue.real

                    + identity_shift

                )


                # ---------------------------------------------
                # Trial circuit
                # ---------------------------------------------

                qc_trial = (
                    ansatz.assign_parameters(

                        result.optimal_parameters

                    )
                )


                # ---------------------------------------------
                # Trial state
                # ---------------------------------------------

                psi_trial = (

                    Statevector
                    .from_instruction(
                        qc_trial
                    )
                    .data

                )


                # ---------------------------------------------
                # Fidelity
                # ---------------------------------------------

                fidelity_trial = (

                    np.abs(

                        np.vdot(
                            psi_exact,
                            psi_trial
                        )

                    ) ** 2

                )


                # ---------------------------------------------
                # Overlap tracking
                # ---------------------------------------------

                if previous_state is None:

                    overlap = 1.0

                else:

                    overlap = (

                        np.abs(

                            np.vdot(

                                previous_state,

                                psi_trial

                            )

                        ) ** 2

                    )


                print(
                    f"Energy   = "
                    f"{E_trial:.12f}"
                )

                print(
                    f"Fidelity = "
                    f"{fidelity_trial:.12f}"
                )

                print(
                    f"Overlap  = "
                    f"{overlap:.12f}"
                )


                # ---------------------------------------------
                # Store candidate
                # ---------------------------------------------

                candidate_states.append({

                    "energy":
                        E_trial,

                    "state":
                        psi_trial,

                    "result":
                        result,

                    "fidelity":
                        fidelity_trial,

                    "overlap":
                        overlap

                })


            # =================================================
            # Minimum energy
            # =================================================

            min_energy = min(

                c["energy"]

                for c in candidate_states

            )


            # =================================================
            # Low-energy manifold
            # =================================================

            filtered_candidates = [

                c

                for c in candidate_states

                if (

                    c["energy"]
                    - min_energy

                ) < energy_tol

            ]


            # =================================================
            # Select maximum overlap
            # =================================================

            best_candidate = max(

                filtered_candidates,

                key=lambda x:
                    x["overlap"]

            )


            # =================================================
            # Final selected state
            # =================================================

            best_energy = (
                best_candidate["energy"]
            )

            best_state = (
                best_candidate["state"]
            )

            best_result = (
                best_candidate["result"]
            )

            best_fidelity = (
                best_candidate["fidelity"]
            )

            best_overlap = (
                best_candidate["overlap"]
            )


            # =================================================
            # Update tracking
            # =================================================

            previous_state = (
                best_state.copy()
            )

            previous_optimal_point = (
                best_result.optimal_point.copy()
            )


        # ====================================================
        # Symmetric + Strong SSB
        # ====================================================

        else:

            print(
                "\nRunning ordinary HVA-VQE...\n"
            )


            # ------------------------------------------------
            # Random initial point
            # ------------------------------------------------

            initial_point = (

                0.01
                * np.random.randn(
                    ansatz.num_parameters
                )

            )


            # ------------------------------------------------
            # VQE
            # ------------------------------------------------

            vqe = VQE(

                estimator=estimator,

                ansatz=ansatz,

                optimizer=optimizer,

                initial_point=initial_point

            )


            result = (
                vqe.compute_minimum_eigenvalue(

                    H_pauli_shifted

                )
            )


            # ------------------------------------------------
            # Physical energy
            # ------------------------------------------------

            best_energy = (

                result.eigenvalue.real

                + identity_shift

            )


            # ------------------------------------------------
            # Optimal HVA circuit
            # ------------------------------------------------

            optimal_circuit = (
                ansatz.assign_parameters(

                    result.optimal_parameters

                )
            )


            # ------------------------------------------------
            # VQE state
            # ------------------------------------------------

            best_state = (

                Statevector
                .from_instruction(
                    optimal_circuit
                )
                .data

            )


            # ------------------------------------------------
            # Fidelity
            # ------------------------------------------------

            best_fidelity = (

                np.abs(

                    np.vdot(

                        psi_exact,

                        best_state

                    )

                ) ** 2

            )


            best_overlap = 1.0


        # ====================================================
        # Final VQE quantities
        # ====================================================

        E_vqe = best_energy

        F = best_fidelity


        # ====================================================
        # Store results
        # ====================================================

        data = {

            "case":
                label,

            "lambda":
                lam,

            "mI2":
                mI2,

            "f":
                f,

            "psi_vqe":
                best_state,

            "E_exact":
                E_exact,

            "E_vqe":
                E_vqe,

            "error":
                abs(
                    E_exact
                    - E_vqe
                ),

            "fidelity":
                F,

            "tracking_overlap":
                best_overlap,

            "layers":
                layers

        }


        case_results.append(
            data
        )

        all_results.append(
            data
        )


        # ====================================================
        # Print final result
        # ====================================================

        print("\nFINAL SELECTED STATE")

        print(
            f"Exact energy     : "
            f"{E_exact:.12f}"
        )

        print(
            f"VQE energy       : "
            f"{E_vqe:.12f}"
        )

        print(
            f"Energy error     : "
            f"{abs(E_exact - E_vqe):.6e}"
        )

        print(
            f"Fidelity         : "
            f"{F:.12f}"
        )

        print(
            f"Tracking overlap : "
            f"{best_overlap:.12f}"
        )

        print(
            f"Layers           : "
            f"{layers}"
        )

        print(
            f"HVA parameters   : "
            f"{ansatz.num_parameters}"
        )

        print(
            f"Pauli terms      : "
            f"{len(H_pauli_shifted)}"
        )


    # ========================================================
    # Case summary
    # ========================================================

    print("\n")

    print(
        "=" * 140
    )

    print(
        f"SUMMARY : {label}"
    )

    print(
        "=" * 140
    )


    print(

        f"{'lambda':<12}"
        f"{'Exact E':<20}"
        f"{'VQE E':<20}"
        f"{'Error':<18}"
        f"{'Fidelity':<18}"
        f"{'TrackOverlap':<18}"
        f"{'Layers':<10}"

    )


    print(
        "=" * 140
    )


    for r in case_results:

        print(

            f"{r['lambda']:<12.4f}"

            f"{r['E_exact']:<20.10f}"

            f"{r['E_vqe']:<20.10f}"

            f"{r['error']:<18.4e}"

            f"{r['fidelity']:<18.10f}"

            f"{r['tracking_overlap']:<18.10f}"

            f"{r['layers']:<10}"

        )



# In[ ]:


# ============================================================
# Plot : Fidelity vs lambda
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Collect data
# ------------------------------------------------------------

cases = {}

for r in all_results:

    label = r['case']

    if label not in cases:

        cases[label] = {
            'lambda': [],
            'fidelity': []
        }

    cases[label]['lambda'].append(r['lambda'])
    cases[label]['fidelity'].append(r['fidelity'])

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
case_colors = {
    "Symmetric": "blue",
    "Weak SSB": "orange",
    "Weak SSB optimised": "green",
    "Strong SSB": "red"
}
#
plt.figure(figsize=(8,6))

for label, data in cases.items():

    plt.plot(
        data['lambda'],
        data['fidelity'],
        marker='o',
        linewidth=2,
        color=case_colors[label],
        label=label
    )

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

plt.xlabel(r'$\lambda$', fontsize=14)

plt.ylabel(r'$F_{loc}$', fontsize=14)

plt.title(
    r'VQE Fidelity vs $\lambda$',
    fontsize=16
)

plt.grid(True)

plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig(
    "HVA_Floc vs lambda different f.pdf",
    bbox_inches='tight'
)

plt.show()


# In[1]:


# ============================================================
# H_c
# ============================================================

def lattice_hc(
    n_sites,
    nq,
    mu=1.0,
    m0_sq=1.0,
    mI2=1.0
):

    # --------------------------------------------------------
    # Embedded field operators
    # --------------------------------------------------------

    Phi, Pi = lattice_operators(
        n_sites,
        nq,
        mu
    )

    # --------------------------------------------------------
    # Initialize coupling Hamiltonian
    # --------------------------------------------------------

    Hc = 0

    # ========================================================
    # Nearest-neighbour coupling
    #
    # H_coupling =
    #     1/2 sum_i (Phi[i+1] - Phi[i])^2
    #
    # Open boundary conditions
    # ========================================================

    for i in range(n_sites - 1):

        dPhi = (
            Phi[i+1] - Phi[i]
        )

        Hc += 0.5 * (
            dPhi @ dPhi
        )

    # ========================================================
    # Mass correction
    #
    # H_mass =
    #     1/2 (m0^2 - mI^2) sum_i Phi[i]^2
    # ========================================================

    for i in range(n_sites):

        Hc += (
            0.5
            * (m0_sq - mI2)
            * (Phi[i] @ Phi[i])
        )

    return Hc


# In[ ]:





# In[ ]:


# ============================================================
# H_c Pauli representation for an N-site lattice
# ============================================================

from qiskit.quantum_info import Operator, SparsePauliOp


def lattice_hc_pauli(
    n_sites,
    nq,
    mu=1.0,
    m0_sq=1.0,
    mI2=1.0
):

    # ========================================================
    # Single-site field operator
    # ========================================================

    Phi, Pi = single_site_field_operators(
        nq,
        mu
    )

    N = 2**nq

    # ========================================================
    # Single-site identity
    # ========================================================

    I_local = identity(
        N,
        format="csr",
        dtype=complex
    )

    # ========================================================
    # Local Phi^2
    # ========================================================

    Phi2 = Phi @ Phi

    # ========================================================
    # Convert local Phi and Phi^2 to Pauli form
    # ========================================================

    Phi_pauli_local = (
        SparsePauliOp.from_operator(
            Operator(
                Phi.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )

    Phi2_pauli_local = (
        SparsePauliOp.from_operator(
            Operator(
                Phi2.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )

    # ========================================================
    # Initialize full H_c
    # ========================================================

    Hc_pauli = None

    # ========================================================
    # Mass-difference term
    #
    # 1/2 (m0^2 - mI^2) sum_i Phi_i^2
    # ========================================================

    mass_coeff = 0.5 * (
        m0_sq - mI2
    )

    if abs(mass_coeff) > 0:

        Hmass_pauli = (
            embed_local_pauli_on_lattice(
                Phi2_pauli_local,
                n_sites,
                nq
            )
        )

        # The embedding above represents Phi^2 on ONE site.
        # We need the sum over all sites.
        Hmass_pauli = None

        for site in range(n_sites):

            pauli_strings = []

            coeffs = []

            for p, c in zip(
                Phi2_pauli_local.paulis,
                Phi2_pauli_local.coeffs
            ):

                local_string = str(p)

                full_string = (
                    "I" * (site * nq)
                    + local_string
                    + "I" * ((n_sites - site - 1) * nq)
                )

                pauli_strings.append(
                    full_string
                )

                coeffs.append(
                    mass_coeff * c
                )

            site_op = SparsePauliOp(
                pauli_strings,
                coeffs=coeffs
            )

            if Hmass_pauli is None:

                Hmass_pauli = site_op

            else:

                Hmass_pauli += site_op

    else:

        Hmass_pauli = SparsePauliOp(
            "I" * (n_sites * nq)
        )

    # ========================================================
    # Nearest-neighbour gradient term
    #
    # 1/2 sum_i (Phi_{i+1} - Phi_i)^2
    #
    # = 1/2 sum_i [
    #       Phi_i^2
    #       + Phi_{i+1}^2
    #       - 2 Phi_i Phi_{i+1}
    #   ]
    # ========================================================

    Hgrad_pauli = None

    for site in range(n_sites - 1):

        # ----------------------------------------------------
        # Phi_i^2
        # ----------------------------------------------------

        for p, c in zip(
            Phi2_pauli_local.paulis,
            Phi2_pauli_local.coeffs
        ):

            local_string = str(p)

            full_string = (
                "I" * (site * nq)
                + local_string
                + "I" * ((n_sites - site - 1) * nq)
            )

            term = SparsePauliOp(
                [full_string],
                coeffs=[0.5 * c]
            )

            if Hgrad_pauli is None:

                Hgrad_pauli = term

            else:

                Hgrad_pauli += term


        # ----------------------------------------------------
        # Phi_{i+1}^2
        # ----------------------------------------------------

        next_site = site + 1

        for p, c in zip(
            Phi2_pauli_local.paulis,
            Phi2_pauli_local.coeffs
        ):

            local_string = str(p)

            full_string = (
                "I" * (next_site * nq)
                + local_string
                + "I" * ((n_sites - next_site - 1) * nq)
            )

            term = SparsePauliOp(
                [full_string],
                coeffs=[0.5 * c]
            )

            Hgrad_pauli += term


        # ----------------------------------------------------
        # - Phi_i Phi_{i+1}
        # ----------------------------------------------------

        for p1, c1 in zip(
            Phi_pauli_local.paulis,
            Phi_pauli_local.coeffs
        ):

            for p2, c2 in zip(
                Phi_pauli_local.paulis,
                Phi_pauli_local.coeffs
            ):

                string1 = str(p1)
                string2 = str(p2)

                full_string = (
                    "I" * (site * nq)
                    + string1
                    + string2
                    + "I" * (
                        (n_sites - site - 2) * nq
                    )
                )

                term = SparsePauliOp(
                    [full_string],
                    coeffs=[-c1 * c2]
                )

                Hgrad_pauli += term

    # ========================================================
    # Total H_c
    # ========================================================

    Hc_pauli = (
        Hgrad_pauli
        + Hmass_pauli
    ).simplify(
        atol=1e-10
    )

    return Hc_pauli


# In[ ]:


# ============================================================
# H_s
# ============================================================

def lattice_hs(
    n_sites,
    nq,
    mu=1.0,
    mI2=1.0,
    m0_sq=1.0,
    lam=1.0,
    f=0.0,
    s=1.0
):

    # --------------------------------------------------------
    # Local Hamiltonian
    # --------------------------------------------------------

    Hloc = lattice_hloc(
        n_sites=n_sites,
        nq=nq,
        mu=mu,
        mI2=mI2,
        lam=lam,
        f=f
    )

    # --------------------------------------------------------
    # Coupling + mass correction
    # --------------------------------------------------------

    Hc = lattice_hc(
        n_sites=n_sites,
        nq=nq,
        mu=mu,
        m0_sq=m0_sq,
        mI2=mI2
    )

    # --------------------------------------------------------
    # Full Hamiltonian
    # --------------------------------------------------------

    Hs = Hloc + s * Hc

    return Hs


# In[2]:


# ============================================================
# Exact ground state
# ============================================================

from scipy.sparse.linalg import eigsh
import numpy as np


def exact_ground_state(
    H,
    tol=1e-10,
    maxiter=None
):

    # --------------------------------------------------------
    # Smallest algebraic eigenvalue
    # --------------------------------------------------------

    eigvals, eigvecs = eigsh(
        H,
        k=1,
        which='SA',
        tol=tol,
        maxiter=maxiter
    )

    # --------------------------------------------------------
    # Ground-state energy
    # --------------------------------------------------------

    E0 = np.real(
        eigvals[0]
    )

    # --------------------------------------------------------
    # Ground-state vector
    # --------------------------------------------------------

    psi0 = eigvecs[:, 0]

    # Explicit normalization
    psi0 = (
        psi0 /
        np.linalg.norm(psi0)
    )

    return E0, psi0


# In[ ]:


# ============================================================
# One Trotter / circuit evolution step
# ============================================================

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate


def trotter_step_circuit(
    n_sites,
    nq,
    dt,
    s,
    Hloc_pauli,
    Hc_pauli
):

    # --------------------------------------------------------
    # Total number of qubits
    # --------------------------------------------------------

    nqubits = n_sites * nq


    # --------------------------------------------------------
    # Instantaneous Hamiltonian
    #
    # H(s) = H_loc + s H_c
    # --------------------------------------------------------

    Hs_pauli = (

        Hloc_pauli
        + s * Hc_pauli

    ).simplify(
        atol=1e-10
    )


    # --------------------------------------------------------
    # Quantum circuit for one evolution step
    #
    # U(dt) = exp[-i H(s) dt]
    # --------------------------------------------------------

    qc = QuantumCircuit(
        nqubits
    )

    qc.append(

        PauliEvolutionGate(
            Hs_pauli,
            time=dt
        ),

        range(nqubits)

    )

    return qc


# In[ ]:


# ============================================================
# Adiabatic circuit evolution
# ============================================================

from qiskit.quantum_info import Statevector


def adiabatic_circuit_evolution(
    initial_state,
    n_sites,
    nq,
    T,
    Hloc_pauli,
    Hc_pauli,
    nsteps=100
):

    # --------------------------------------------------------
    # Total number of qubits
    # --------------------------------------------------------

    nqubits = n_sites * nq

    # --------------------------------------------------------
    # Time step
    # --------------------------------------------------------

    dt = T / nsteps

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    psi = Statevector(initial_state)

    # --------------------------------------------------------
    # Time evolution
    # --------------------------------------------------------

    for step in range(nsteps):

        # ----------------------------------------------------
        # Midpoint of the current time interval
        # ----------------------------------------------------

        s = (step + 0.5) / nsteps

        # ----------------------------------------------------
        # One instantaneous evolution circuit
        #
        # H(s) = H_loc + s H_c
        # ----------------------------------------------------

        qc = trotter_step_circuit(

            n_sites=n_sites,

            nq=nq,

            dt=dt,

            s=s,

            Hloc_pauli=Hloc_pauli,

            Hc_pauli=Hc_pauli
        )

        # ----------------------------------------------------
        # Evolve state
        # ----------------------------------------------------

        psi = psi.evolve(qc)

    return psi.data


# # Starting of m0^2=1

# In[3]:


# ============================================================
# T scan values
# ============================================================

T_values = [

    1, 2, 5, 10, 20, 30, 40, 50, 75, 100,
    125, 150, 175, 200, 225, 250, 275, 300,
    400, 500, 600, 700, 800, 900,
    1000, 1010, 1020, 1030, 1040, 1050,
    1060, 1070, 1080, 1090, 1100,
    1200, 1300, 1400, 1500,
    2000,
    2100, 2110, 2120, 2130, 2140,
    2150, 2160, 2170, 2180, 2190,
    2200, 2300, 2400, 2500,
    2600, 2700, 2800, 2900, 3000

]
import sys

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python 16d_6sites_HVA-VQE_modified_time_evolution-layers2.py <m0_sq>"
    )

m0_sq = float(sys.argv[1])

print("=" * 80)
print(f"User supplied m0^2 = {m0_sq}")
print("=" * 80)
target_fidelity = 0.99


# ============================================================
# Storage
# ============================================================

adiabatic_results = []


# ============================================================
# MAIN LOOP
# ============================================================

for data in all_results:

    label   = data['case']
    lam     = data['lambda']
    mI2     = data['mI2']
    f       = data['f']
    psi_vqe = data['psi_vqe']


    print("\n")
    print("=" * 80)

    print(f"CASE   : {label}")
    print(f"lambda : {lam:.3f}")

    print("=" * 80)


    # ========================================================
    # Build H_loc
    #
    # This is the same H_loc on which VQE was performed.
    # ========================================================

    (
        Hpi_sparse,
        Hphi2_sparse,
        Hphi4_sparse,
        Hf_sparse,
        Hloc_sparse

    ) = lattice_hloc_decomposed(

        n_sites=n_sites,

        nq=nq_site,

        mu=mu,

        mI2=mI2,

        lam=lam,

        f=f

    )


    # ========================================================
    # Exact ground state of H_loc
    #
    # This is NOT the final adiabatic ground state.
    # It is only the exact reference for checking VQE.
    # ========================================================

    E_exact_loc, psi_exact_loc = (
        exact_ground_state(
            Hloc_sparse,
            tol=1e-10
        )
    )


    print(
        f"H_loc exact energy = "
        f"{E_exact_loc:.12f}"
    )


    # ========================================================
    # Build H_loc Pauli representation
    #
    # Convert LOCAL 16 x 16 pieces to Pauli form and
    # embed them onto the 6-site / 24-qubit lattice.
    # ========================================================

    Phi_local, Pi_local = (
        single_site_field_operators(
            nq_site,
            mu
        )
    )

    Phi2_local = (
        Phi_local @ Phi_local
    )

    Phi4_local = (
        Phi2_local @ Phi2_local
    )

    Pi2_local = (
        Pi_local @ Pi_local
    )


    hpi_local = (
        0.5 * Pi2_local
    )

    hphi2_local = (
        0.5 * mI2 * Phi2_local
    )

    hphi4_local = (
        (lam / 24.0) * Phi4_local
    )

    hf_local = (
        f * Phi_local
    )


    # ========================================================
    # Convert LOCAL pieces to Pauli operators
    # ========================================================

    Hpi_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hpi_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hphi2_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hphi2_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hphi4_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hphi4_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hf_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hf_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    # ========================================================
    # Embed local Pauli pieces onto 6 sites
    # ========================================================

    Hpi_pauli = embed_local_pauli_on_lattice(
        Hpi_local_pauli,
        n_sites,
        nq_site
    )

    Hphi2_pauli = embed_local_pauli_on_lattice(
        Hphi2_local_pauli,
        n_sites,
        nq_site
    )

    Hphi4_pauli = embed_local_pauli_on_lattice(
        Hphi4_local_pauli,
        n_sites,
        nq_site
    )

    Hf_pauli = embed_local_pauli_on_lattice(
        Hf_local_pauli,
        n_sites,
        nq_site
    )


    # ========================================================
    # Total H_loc in Pauli form
    # ========================================================

    Hloc_pauli = (

        Hpi_pauli
        + Hphi2_pauli
        + Hphi4_pauli
        + Hf_pauli

    ).simplify(
        atol=1e-10
    )


    # ========================================================
    # Build H_c Pauli representation
    #
    # H_c is NOT used in VQE.
    # It enters only during adiabatic evolution.
    # ========================================================

    Hc_pauli = lattice_hc_pauli(

        n_sites=n_sites,

        nq=nq_site,

        mu=mu,

        m0_sq=m0_sq,

        mI2=mI2

    )


    # ========================================================
    # Final Hamiltonian
    #
    # H_s(s=1) = H_loc + H_c
    #
    # This is used only for the final exact reference state.
    # ========================================================

    Hs_final = lattice_hs(

        n_sites=n_sites,

        nq=nq_site,

        mu=mu,

        mI2=mI2,

        m0_sq=m0_sq,

        lam=lam,

        f=f,

        s=1.0

    )


    # ========================================================
    # Exact ground state of FINAL Hamiltonian
    # ========================================================

    E_exact_final, psi_exact_final = (
        exact_ground_state(

            Hs_final,

            tol=1e-10

        )
    )


    print(
        f"Final exact energy = "
        f"{E_exact_final:.12f}"
    )


    # ========================================================
    # Scan T
    # ========================================================

    Tmin_found = None

    best_fidelity = 0.0


    for T in T_values:

        # ====================================================
        # Adiabatic evolution
        #
        # Initial state:
        #     psi_vqe
        #
        # Evolution:
        #     H(s) = H_loc + s H_c
        #
        # Hloc_pauli and Hc_pauli were constructed ONCE
        # above and are reused for every Trotter step.
        # ====================================================

        psi_T = adiabatic_circuit_evolution(

            initial_state=psi_vqe,

            n_sites=n_sites,

            nq=nq_site,

            T=T,

            Hloc_pauli=Hloc_pauli,

            Hc_pauli=Hc_pauli,

            nsteps=100

        )


        # ====================================================
        # Fidelity with FINAL exact ground state
        # ====================================================

        F = fidelity(

            psi_exact_final,

            psi_T

        )


        best_fidelity = max(

            best_fidelity,

            F

        )


        print(

            f"T = {T:<5} "
            f"Fidelity = {F:.8f}"

        )


        # ----------------------------------------------------
        # First T satisfying target fidelity
        # ----------------------------------------------------

        if F >= target_fidelity:

            Tmin_found = T

            break


    # ========================================================
    # Diagnostics
    # ========================================================

    P0 = None
    P1 = None
    P01 = None
    final_gap = None


    # ========================================================
    # If target fidelity was not reached
    # ========================================================

    if Tmin_found is None:

        print("\nFAILED POINT DIAGNOSTICS")


        # ----------------------------------------------------
        # Use largest T as diagnostic point
        # ----------------------------------------------------

        T_diagnostic = T_values[-1]


        psi_T = adiabatic_circuit_evolution(

            initial_state=psi_vqe,

            n_sites=n_sites,

            nq=nq_site,

            T=T_diagnostic,

            Hloc_pauli=Hloc_pauli,

            Hc_pauli=Hc_pauli,

            nsteps=100

        )


        # ----------------------------------------------------
        # Three lowest eigenstates of final H_s
        # ----------------------------------------------------

        evals, evecs = eigsh(

            Hs_final,

            k=3,

            which='SA'

        )


        # ----------------------------------------------------
        # Sort eigenvalues
        # ----------------------------------------------------

        idx = np.argsort(evals)

        evals = evals[idx]

        evecs = evecs[:, idx]


        # ----------------------------------------------------
        # Ground and first excited states
        # ----------------------------------------------------

        psi0 = evecs[:, 0]

        psi1 = evecs[:, 1]


        # ----------------------------------------------------
        # Populations
        # ----------------------------------------------------

        P0 = np.abs(

            np.vdot(
                psi0,
                psi_T
            )

        )**2


        P1 = np.abs(

            np.vdot(
                psi1,
                psi_T
            )

        )**2


        P01 = P0 + P1


        # ----------------------------------------------------
        # Final energy gap
        # ----------------------------------------------------

        final_gap = (
            evals[1] - evals[0]
        )


        print(
            f"E1-E0  = {final_gap:.8e}"
        )

        print(
            f"P0     = {P0:.8f}"
        )

        print(
            f"P1     = {P1:.8f}"
        )

        print(
            f"P0+P1  = {P01:.8f}"
        )


    # ========================================================
    # Store results
    # ========================================================

    adiabatic_results.append({

        'case': label,

        'lambda': lam,

        'mI2': mI2,

        'f': f,

        # Initial state obtained from VQE on H_loc
        'psi_vqe': psi_vqe,

        'Tmin': Tmin_found,

        'best_fidelity': best_fidelity,

        'P0': P0,

        'P1': P1,

        'P01': P01,

        'final_gap': final_gap

    })


    print(

        f"\nMinimum T for Fidelity >= 0.99 : "
        f"{Tmin_found}"

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 180)

print("FINAL ADIABATIC SUMMARY")

print("=" * 180)

print(

    f"{'Case':<15}"
    f"{'lambda':<10}"
    f"{'T_min':<10}"
    f"{'Best Fidelity':<18}"
    f"{'P0':<12}"
    f"{'P1':<12}"
    f"{'P0+P1':<12}"

)

print("=" * 180)

for r in adiabatic_results:

    print(

        f"{r['case']:<15}"

        f"{r['lambda']:<10.3f}"

        f"{str(r['Tmin']):<10}"

        f"{r['best_fidelity']:<18.10f}"

        f"{str(None if r['P0'] is None else round(r['P0'], 6)):<12}"

        f"{str(None if r['P1'] is None else round(r['P1'], 6)):<12}"

        f"{str(None if r['P01'] is None else round(r['P01'], 6)):<12}"

    )


# In[ ]:


# ============================================================
# Plot T_min vs lambda
# ============================================================

import matplotlib.pyplot as plt

# ============================================================
# Separate data by case
# ============================================================

case_data = {}

for r in adiabatic_results:

    case = r['case']

    if case not in case_data:

        case_data[case] = {

            'lambda': [],

            'Tmin': []
        }

    case_data[case]['lambda'].append(

        r['lambda']
    )

    case_data[case]['Tmin'].append(

        r['Tmin']
    )

# ============================================================
# Plot
# ============================================================
case_colors = {
    "Symmetric": "blue",
    "Weak SSB": "orange",
    "Weak SSB optimised": "green",
    "Strong SSB": "red"
}
#

plt.figure(figsize=(8,6))

for case, d in case_data.items():

    plt.plot(

        d['lambda'],

        d['Tmin'],

        marker='o',

        linewidth=2,

        color=case_colors[case],

        label=case
    )

# ============================================================
# Labels
# ============================================================

plt.xlabel(r'$\lambda$', fontsize=14)

plt.ylabel(r'$T_{\min}$', fontsize=14)

plt.title(

    rf'Minimum Adiabatic Time $T_{\min}$ '
    rf'for Fidelity $\geq 0.99$, $m_{0}^{2}={m0_sq}$',

    fontsize=14
)

plt.grid(True)

plt.legend()

plt.tight_layout()
plt.savefig(
    f"HVA_adiabatic time_vs_lambda_m0sq_{m0_sq}.pdf",
    bbox_inches='tight'
)
#plt.show()


# In[ ]:


# ============================================================
# Order parameter operator
# ============================================================

Phi, Pi = lattice_operators(
    n_sites=n_sites,
    nq=nq_site,
    mu=mu
)

# ============================================================
# Average field over all lattice sites
# ============================================================

Phi_avg = sum(Phi) / n_sites


# In[ ]:


print(adiabatic_results[0].keys())


# ============================================================
# <phi> from evolved state
# ============================================================

M_sym = []

M_weak_opt = []

M_strong = []

lambda_sym = []

lambda_weak_opt = []

lambda_strong = []


for r in adiabatic_results:

    case = r['case']

    if case not in [

        "Symmetric",

        "Weak SSB optimised",

        "Strong SSB"

    ]:

        continue


    lam = r['lambda']

    Tmin = r['Tmin']


    # ========================================================
    # Skip if no Tmin was found
    # ========================================================

    if Tmin is None:
        continue

    if np.isnan(Tmin):
        continue


    mI2 = r['mI2']

    f = r['f']


    # ========================================================
    # Find corresponding VQE result
    # ========================================================

    match = next(

        x for x in all_results

        if (

            x['case'] == case

            and np.isclose(
                x['lambda'],
                lam
            )

        )

    )


    psi_vqe = match['psi_vqe']


    # ========================================================
    # Build H_loc Pauli representation
    #
    # This is the same H_loc used for VQE.
    # ========================================================

    Phi_local, Pi_local = (
        single_site_field_operators(
            nq_site,
            mu
        )
    )

    Phi2_local = (
        Phi_local @ Phi_local
    )

    Phi4_local = (
        Phi2_local @ Phi2_local
    )

    Pi2_local = (
        Pi_local @ Pi_local
    )


    hpi_local = (
        0.5 * Pi2_local
    )

    hphi2_local = (
        0.5 * mI2 * Phi2_local
    )

    hphi4_local = (
        (lam / 24.0) * Phi4_local
    )

    hf_local = (
        f * Phi_local
    )


    # ========================================================
    # Convert local pieces to Pauli form
    # ========================================================

    Hpi_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hpi_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hphi2_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hphi2_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hphi4_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hphi4_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    Hf_local_pauli = (
        SparsePauliOp.from_operator(
            Operator(
                hf_local.toarray()
            )
        ).simplify(
            atol=1e-10
        )
    )


    # ========================================================
    # Embed onto 6-site / 24-qubit lattice
    # ========================================================

    Hpi_pauli = embed_local_pauli_on_lattice(
        Hpi_local_pauli,
        n_sites,
        nq_site
    )

    Hphi2_pauli = embed_local_pauli_on_lattice(
        Hphi2_local_pauli,
        n_sites,
        nq_site
    )

    Hphi4_pauli = embed_local_pauli_on_lattice(
        Hphi4_local_pauli,
        n_sites,
        nq_site
    )

    Hf_pauli = embed_local_pauli_on_lattice(
        Hf_local_pauli,
        n_sites,
        nq_site
    )


    # ========================================================
    # Total H_loc Pauli operator
    # ========================================================

    Hloc_pauli = (

        Hpi_pauli
        + Hphi2_pauli
        + Hphi4_pauli
        + Hf_pauli

    ).simplify(
        atol=1e-10
    )


    # ========================================================
    # Build H_c Pauli representation
    #
    # H_c is used only for adiabatic evolution.
    # ========================================================

    Hc_pauli = lattice_hc_pauli(

        n_sites=n_sites,

        nq=nq_site,

        mu=mu,

        m0_sq=m0_sq,

        mI2=mI2

    )


    # ========================================================
    # Adiabatic evolution
    #
    # Initial state:
    #     VQE ground-state approximation of H_loc
    #
    # Evolution:
    #     H(s) = H_loc + s H_c
    # ========================================================

    psi_final = adiabatic_circuit_evolution(

        initial_state=psi_vqe,

        n_sites=n_sites,

        nq=nq_site,

        T=Tmin,

        Hloc_pauli=Hloc_pauli,

        Hc_pauli=Hc_pauli,

        nsteps=100

    )


    # ========================================================
    # Order parameter
    #
    # Phi_avg = (1/N) sum_i Phi_i
    # ========================================================

    M = np.real(

        np.vdot(

            psi_final,

            Phi_avg @ psi_final

        )

    )


    # ========================================================
    # Store by case
    # ========================================================

    if case == "Symmetric":

        M_sym.append(M)

        lambda_sym.append(lam)


    elif case == "Weak SSB optimised":

        M_weak_opt.append(M)

        lambda_weak_opt.append(lam)


    elif case == "Strong SSB":

        M_strong.append(M)

        lambda_strong.append(lam)


# In[ ]:


# ============================================================
# λc from inflection point
# ============================================================

# ============================================================
# First derivatives
# ============================================================

dM_sym = np.gradient(
    M_sym,
    lambda_sym
)

dM_weak = np.gradient(
    M_weak_opt,
    lambda_weak_opt
)

dM_strong = np.gradient(
    M_strong,
    lambda_strong
)


# ============================================================
# Second derivatives
# ============================================================

d2M_sym = np.gradient(
    dM_sym,
    lambda_sym
)

d2M_weak = np.gradient(
    dM_weak,
    lambda_weak_opt
)

d2M_strong = np.gradient(
    dM_strong,
    lambda_strong
)


# ============================================================
# Third derivatives
# ============================================================

d3M_sym = np.gradient(
    d2M_sym,
    lambda_sym
)

d3M_weak = np.gradient(
    d2M_weak,
    lambda_weak_opt
)

d3M_strong = np.gradient(
    d2M_strong,
    lambda_strong
)


# ============================================================
# Find λc from zero crossing of d²M/dλ²
# ============================================================

def find_lambda_c(
    lams,
    d2M,
    d3M
):

    for i in range(
        len(lams) - 1
    ):

        # ----------------------------------------------------
        # Look for sign change in second derivative
        # ----------------------------------------------------

        if (
            d2M[i] * d2M[i+1]
            <= 0
        ):

            # ------------------------------------------------
            # Select the physically relevant inflection point
            # using the sign of the third derivative
            # ------------------------------------------------

            if d3M[i] < 0:

                x1 = lams[i]
                x2 = lams[i+1]

                y1 = d2M[i]
                y2 = d2M[i+1]

                # --------------------------------------------
                # Linear interpolation
                # --------------------------------------------

                if y2 != y1:

                    return (
                        x1
                        - y1 * (x2 - x1)
                        / (y2 - y1)
                    )

    return np.nan


# ============================================================
# Critical lambda values
# ============================================================

lamc_M_sym = find_lambda_c(
    lambda_sym,
    d2M_sym,
    d3M_sym
)

lamc_M_weak = find_lambda_c(
    lambda_weak_opt,
    d2M_weak,
    d3M_weak
)

lamc_M_strong = find_lambda_c(
    lambda_strong,
    d2M_strong,
    d3M_strong
)


# ============================================================
# Print results
# ============================================================

print(
    "\nCritical lambda from inflection point:"
)

print(
    f"Symmetric           : "
    f"{lamc_M_sym:.8f}"
)

print(
    f"Weak SSB optimised  : "
    f"{lamc_M_weak:.8f}"
)

print(
    f"Strong SSB          : "
    f"{lamc_M_strong:.8f}"
)


# In[1]:


########## Routine to plot d2M/d\lambda2


# ============================================================
# Plot : d²<phi>/dλ² vs λ
# ============================================================

plt.figure(figsize=(8,6))

line1, = plt.plot(
    lambda_sym,
    d2M_sym,
    'o-',
    label='Symmetric',
    color='b'
)

line2, = plt.plot(
    lambda_weak_opt,
    d2M_weak,
    's-',
    label='Weak SSB optimised',
    color='g'
)

line3, = plt.plot(
    lambda_strong,
    d2M_strong,
    '^-',
    label='Strong SSB',
    color='r'
)

# λc positions from inflection-point analysis

plt.axvline(
    lamc_M_sym,
    linestyle='--',
    color=line1.get_color(),
    alpha=0.8
)

plt.axvline(
    lamc_M_weak,
    linestyle='--',
    color=line2.get_color(),
    alpha=0.8
)

plt.axvline(
    lamc_M_strong,
    linestyle='--',
    color=line3.get_color(),
    alpha=0.8
)

# Zero line

plt.axhline(
    0,
    color='k',
    linestyle=':',
    linewidth=1.5
)

plt.xlabel(r'$\lambda$')

plt.ylabel(
    r'$d^2\langle\phi\rangle/d\lambda^2$'
)

plt.title(
    rf'Inflection-point analysis $m_0^2={m0_sq}$'
)

plt.grid(True)

plt.legend()

plt.tight_layout()

#plt.show()
plt.savefig(
    f"d2phi_dlambda2_vs_lambda_m0sq_{m0_sq}.pdf",
    bbox_inches='tight'
)


# In[ ]:


# ============================================================
# Plot : <phi> vs lambda
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(

    lambda_sym,

    M_sym,

    'o-',

    label='Symmetric',
    color='b'
)

plt.plot(

    lambda_weak_opt,

    M_weak_opt,

    's-',

    label='Weak SSB optimised',
    color='g'
)

plt.plot(

    lambda_strong,

    M_strong,

    '^-',

    label='Strong SSB',
    color='r'
)

plt.xlabel(r'$\lambda$')

plt.ylabel(r'$\langle \phi \rangle$')

plt.title('Order Parameter')

plt.grid(True)

plt.legend()

plt.tight_layout()

#plt.show()
plt.savefig(
    f"phi_vs_lambda_m0sq_{m0_sq}.pdf",
    bbox_inches='tight'
)


# In[ ]:


# ============================================================
# Plot : d<phi>/dlambda vs lambda
# ============================================================

plt.figure(figsize=(8,6))

line1, = plt.plot(
    lambda_sym,
    dM_sym,
    'o-',
    label='Symmetric',
    color='b'
)

line2, = plt.plot(
    lambda_weak_opt,
    dM_weak,
    's-',
    label='Weak SSB optimised',
    color='g'
)

line3, = plt.plot(
    lambda_strong,
    dM_strong,
    '^-',
    label='Strong SSB',
    color='r'
)

# Critical lambdas

plt.axvline(
    lamc_M_sym,
    linestyle='--',
    color=line1.get_color(),
    alpha=0.8
)

plt.axvline(
    lamc_M_weak,
    linestyle='--',
    color=line2.get_color(),
    alpha=0.8
)

plt.axvline(
    lamc_M_strong,
    linestyle='--',
    color=line3.get_color(),
    alpha=0.8
)

plt.axhline(
    0,
    color='k',
    linestyle=':',
    linewidth=1
)

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$d\langle\phi\rangle/d\lambda$')

plt.title(
    rf'Susceptibility-like quantity $d\langle\phi\rangle/d\lambda$, $m_{0}^{2}={m0_sq}$'
)

plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(
    f"dphi_dlambda_vs_lambda_m0sq_{m0_sq}.pdf",
    bbox_inches='tight'
)


# # Start of m0^2=2

# In[ ]:





# In[ ]:





# ## m0^2=-1
