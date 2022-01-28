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
# pylint: disable=invalid-name

"""Tests for schrieffer_wolff.py."""

import numpy as np

from qiskit_dynamics.perturbation import schrieffer_wolff, Multiset
from qiskit_dynamics.perturbation.schrieffer_wolff import solve_commutator_projection, commutator

from ..common import QiskitDynamicsTestCase


class Testschrieffer_wolff(QiskitDynamicsTestCase):
    """Test schrieffer_wolff_recursive_construction function."""

    def test_simple_case(self):
        """Test a simple case with Pauli operators."""
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Z = np.array([[1.0, 0.0], [0.0, -1.0]])

        H0 = Z
        H1 = X

        # manually compute SW transformation up to 5th order
        rhs1 = H1
        S1 = solve_commutator_projection(H0, rhs1)

        rhs2 = commutator(S1, 0.5 * commutator(S1, H0) + H1)
        S2 = solve_commutator_projection(H0, rhs2)

        rhs3 = (
            commutator(S2, 0.5 * commutator(S1, H0) + H1)
            + commutator(S1, 0.5 * commutator(S2, H0))
            + commutator(S1, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
        )
        S3 = solve_commutator_projection(H0, rhs3)

        rhs4 = (commutator(S3, 0.5 * commutator(S1, H0) + H1)
                + commutator(S2, 0.5 * commutator(S2, H0))
                + commutator(S1, 0.5 * commutator(S3, H0))
                + commutator(S2, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
                + commutator(S1, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
                + commutator(S1, commutator(S1, commutator(S2, H0) / 6))
                + commutator(S1, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6))))
        S4 = solve_commutator_projection(H0, rhs4)

        rhs5 = (commutator(S4, 0.5 * commutator(S1, H0) + H1)
                + commutator(S1, 0.5 * commutator(S4, H0))
                + commutator(S3, 0.5 * commutator(S2, H0))
                + commutator(S2, 0.5 * commutator(S3, H0))
                + commutator(S3, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
                + commutator(S1, commutator(S3, commutator(S1, H0) / 6 + H1 / 2))
                + commutator(S1, commutator(S1, commutator(S3, H0) / 6))
                + commutator(S2, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
                + commutator(S2, commutator(S1, commutator(S2, H0) / 6))
                + commutator(S1, commutator(S2, commutator(S2, H0) / 6))
                + commutator(S2, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
                + commutator(S1, commutator(S2, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
                + commutator(S1, commutator(S1, commutator(S2, commutator(S1, H0) / 24 + H1 / 6)))
                + commutator(S1, commutator(S1, commutator(S1, commutator(S2, H0) / 24)))
                + commutator(S1, commutator(S1, commutator(S1, commutator(S1, commutator(S1, H0) / (24 * 5) + H1 / 24)))))
        S5 = solve_commutator_projection(H0, rhs5)

        expected = np.array([S1, S2, S3, S4, S5])
        output = schrieffer_wolff(H0, perturbations=[H1], expansion_order=5).array_coefficients
        self.assertAllClose(expected, output)
