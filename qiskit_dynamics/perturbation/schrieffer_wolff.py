# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Schrieffer-Wolff perturbation computation.
"""

from typing import List, Optional

import numpy as np
from scipy.linalg import solve_sylvester

from qiskit_dynamics.perturbation import ArrayPolynomial, Multiset
from qiskit_dynamics.perturbation.multiset import get_all_submultisets
from qiskit_dynamics.perturbation.solve_lmde_perturbation import merge_expansion_order_indices

def schrieffer_wolff(H0: np.ndarray,
                     perturbations: List[np.ndarray],
                     expansion_order: Optional[int] = None,
                     expansion_labels: Optional[List[Multiset]] = None,
                     perturbation_labels: Optional[List[Multiset]] = None,
                     tol: Optional[float] = 1e-15) -> ArrayPolynomial:
    """Construct truncated multi-variable Schrieffer-Wolff transformation.

    Not sure what the correct output should be but the ``ArrayPolynomial`` will contain
    all relevant information.

    Also, for now we assume H0 is non-degenerate for initial development, but we
    should add a "degeneracy_tol" argument for detecting degeneracies when we add this case.
    More generally we could think about allowing the user to choose the projection,
    so they could e.g. choose a block-diagonal structure even if there are no degeneracies.

    Further notes to self:
        - I wrote this initially thinking it could be vectorized, but I think solve_sylvester
          only works for 2d arrays. Also I just realized that even if solve_sylvester
          was vectorized I haven't written the solve_commutator_projection function to
          be vectorized (due to the conditionals). May want to consider modifying things so
          that it explicitly works with 2d arrays, otherwise the "vectorized" stuff is just
          confusing to read
    """

    ##################################################################################################
    # To do: add validation. For validating the expansion_order/labels args we could
    # move the validation for both solve_lmde_perturbation and this function into
    # merge_expansion_order_indices (and maybe also move this function into multiset.py)
    # Also: validate that H0 is diagonal
    ##################################################################################################

    perturbations = np.array(perturbations)
    mat_shape = perturbations[0].shape

    if perturbation_labels is None:
        perturbation_labels = [Multiset({k: 1}) for k in range(len(perturbations))]

    # get all requested terms in the expansion
    expansion_labels = merge_expansion_order_indices(expansion_order,
                                                     expansion_labels,
                                                     perturbation_labels,
                                                     symmetric=True)
    expansion_labels = get_all_submultisets(expansion_labels)

    # construct labels for recursive computation
    recursive_labels = []
    for label in expansion_labels:
        for k in range(len(label), 0, -1):
            recursive_labels.append((label, k))

    # initialize arrays used to store results
    recursive_A = np.zeros((len(recursive_labels),) + mat_shape, dtype=complex)
    recursive_B = np.zeros((len(recursive_labels),) + mat_shape, dtype=complex)
    expansion_terms = np.zeros((len(expansion_labels),) + mat_shape, dtype=complex)

    # right hand side storage
    rhs_mat = None

    # recursively compute all matrices
    for (recursive_idx, (expansion_label, recursion_order)) in enumerate(recursive_labels):
        expansion_idx = expansion_labels.index(expansion_label)

        if recursion_order == len(expansion_label):
            rhs_mat = np.zeros(mat_shape, dtype=complex)

        # if recursion_order == 1, need to compute the expansion_term for expansion_label
        # and initialize recursion base cases
        if recursion_order == 1:
            # if expansion_label in perturbation labels, add to rhs and initialize
            # recursive_B base case
            if expansion_label in perturbation_labels:
                rhs_mat = rhs_mat + perturbations[expansion_idx]
                recursive_B[recursive_idx] = perturbations[expansion_idx]
            # solve for expansion term
            expansion_terms[expansion_idx] = solve_commutator_projection(H0, rhs_mat, tol=tol)

            # initialize recursive_A base case
            recursive_A[recursive_idx] = commutator(expansion_terms[expansion_idx], H0)
        else:
            # get all 2-fold partitions
            submultisets, complements = expansion_label.submultisets_and_complements()
            for submultiset, complement in zip(submultisets, complements):
                if len(complement) >= recursion_order - 1:
                    SI = expansion_terms[expansion_labels.index(submultiset)]
                    recursive_lower_idx = recursive_labels.index((complement, recursion_order - 1))
                    recursive_A[recursive_idx] += commutator(SI, recursive_A[recursive_lower_idx])
                    recursive_B[recursive_idx] += commutator(SI, recursive_B[recursive_lower_idx])

            recursive_A[recursive_idx] = recursive_A[recursive_idx] / recursion_order
            recursive_B[recursive_idx] = recursive_B[recursive_idx] / (recursion_order - 1)
            rhs_mat = rhs_mat + recursive_A[recursive_idx] + recursive_B[recursive_idx]

    return ArrayPolynomial(array_coefficients=expansion_terms,
                           monomial_multisets=expansion_labels)


def solve_commutator_projection(H0, rhs_mat, tol=1e-15):
    """Solve [H0, X] = rhs_mat assuming H0 is diagonal.

    For now this assumes that the projection is onto off-diagonal, but we can potentially
    modify this later.
    """
    rhs_mat = project_off_diagonal(rhs_mat)

    # if rhs_mat is zero after projection, return 0
    if np.max(np.abs(rhs_mat)) < tol:
        return np.zeros(H0.shape, dtype=complex)

    return solve_sylvester(H0, -H0, rhs_mat)


def project_off_diagonal(A):
    """Set diagonal elements to 0. Assumes A.ndim >= 2. If A.ndim > 2, treats array as a
    multidimensional array of matrices, with last two dimensiosn corresponding to the matrices
    whose diagonal is to be set to 0.
    """
    A = A.copy()
    k = min(A.shape[-2:])
    A[..., range(k), range(k)] = 0.
    return A


def commutator(A, B):
    """Matrix commutator."""
    return A @ B - B @ A
