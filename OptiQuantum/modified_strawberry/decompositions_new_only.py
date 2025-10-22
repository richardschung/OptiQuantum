from itertools import groupby
from collections import defaultdict

import numpy as np
from scipy.linalg import block_diag, sqrtm, polar, schur
from thewalrus.quantum import adj_scaling
from thewalrus.symplectic import sympmat, xpxp_to_xxpp
from strawberryfields.decompositons import nullMZ, mach_zehnder 

def triangular_MZ(V, tol=1e-11):
    """Based on Xanadu Strawberryfields"""
    r"""Triangular decomposition of a unitary matrix due to Reck et al.

    See :cite:`reck1994` for more details and :cite:`clements2016` for details on notation.

    Args:
        V (array[complex]): unitary matrix of size ``n_size``
        tol (float): the tolerance used when checking if the matrix is unitary:
            :math:`|VV^\dagger-I| \leq` tol

    Returns:
        tuple[array]: returns a tuple of the form ``(tlist,np.diag(localV), None)``
            where:

            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary applied at the beginning of circuit
    """
    localV = V
    (nsize, _) = localV.shape

    if not np.allclose(V @ V.conj().T, np.identity(nsize), atol=tol, rtol=0):
        raise ValueError("The input matrix is not unitary")

    tlist = []
    for i in range(nsize - 2, -1, -1):
        for j in range(i + 1):
            tlist.append(nullMZ(nsize - j - 1, nsize - i - 2, localV))
            localV = mach_zehnder(*tlist[-1]) @ localV

    return list(reversed(tlist)), np.diag(localV), None

def triangular_symmetric(V, tol=1e-11):
    """Based on Xanadu Strawberryfields"""
    r"""Decomposition of a unitary into an array of symmetric beamsplitters.

    This decomposition starts with the output from :func:`~.triangular_MZ`
    and performs the equivalent of :func:`~.rectangular_phase_end` by placing all the
    local phase shifts after the interferometers.

    If the Mach-Zehnder unitaries are represented as M and the local phase shifts as D, the new
    parameters to shift the local phases to the end are calculated such that

    .. math::

       M^{-1} D = D_{\mathrm{new}} M_{\mathrm{new}}

    Args:
        V (array): unitary matrix of size n_size
        tol (int): the number of decimal places to use when determining
          whether the matrix is unitary

    Returns:
        tuple[array]: returns a tuple of the form ``(tlist,np.diag(localV), None)``
            where:

            * ``tlist``: list containing ``[n, m, internal_phase, external_phase, n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary matrix to be applied at the end of circuit
            * ``None``: the value ``None``, in order to make the return
              signature identical to :func:`~.rectangular`
    """
    tlist, diags, __ = triangular_MZ(V, tol)
    new_tlist, new_diags = [], diags.copy()

    # Push each beamsplitter through the diagonal unitary
    for i in tlist:
        em, en = int(i[0]), int(i[1])
        alpha, beta = np.angle(new_diags[em]), np.angle(new_diags[en])
        phi_i, phi_e = i[2], i[3]

        # The new parameters required for D', MZ' st. MZ^(-1)D = D'MZ'

        new_phi_e = (alpha - beta) % (2 * np.pi)
        new_alpha = (beta - phi_e - phi_i + np.pi) % (2 * np.pi)
        new_beta = (beta - phi_i + np.pi) % (2 * np.pi)
        new_phi_i = phi_i % (2 * np.pi)
        # repeat modulo operations , otherwise the input unitary
        # numpy.identity(20) yields an external_phase of exactly 2 * pi
        new_phi_i %= 2 * np.pi
        new_phi_e %= 2 * np.pi

        new_i = [i[0], i[1], new_phi_i, new_phi_e, i[4]]
        new_diags[em], new_diags[en] = np.exp(1j * new_alpha), np.exp(1j * new_beta)

        new_tlist = new_tlist + [new_i]

    return new_tlist, new_diags, None
