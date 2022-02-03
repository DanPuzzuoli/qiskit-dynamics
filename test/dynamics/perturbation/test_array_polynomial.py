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

"""Tests for functions in ArrayPolynomial."""

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.perturbation import Multiset, ArrayPolynomial

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit
    from jax import grad, jit
except ImportError:
    pass


class TestArrayPolynomialAlgebra(QiskitDynamicsTestCase):
    """Test algebraic operations on ArrayPolynomials."""

    def test_addition_validation_error(self):
        """Test shape broadcasting failure."""
        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(1, 4, 6) + 1j * np.random.rand(1, 4, 6),
            monomial_labels=[[0]],
            constant_term=np.random.rand(4, 6) + 1j * np.random.rand(4, 6),
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )

        with self.assertRaisesRegex(QiskitError, "broadcastable"):
            ap1 + ap2

    def test_addition_simple(self):
        """Test basic addition."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )
        result = ap1 + ap2

        self.assertAllClose(
            result.array_coefficients, ap1.array_coefficients + ap2.array_coefficients
        )
        self.assertTrue(result.monomial_labels == ap1.monomial_labels)
        self.assertAllClose(result.constant_term, ap1.constant_term + ap2.constant_term)

    def test_addition_non_overlapping_labels(self):
        """Test non-overlapping labels."""
        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [3], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )
        result = ap1 + ap2

        expected_coefficients = np.array([ap1.array_coefficients[0] + ap2.array_coefficients[0],
                                          ap1.array_coefficients[1],
                                          ap1.array_coefficients[2],
                                          ap2.array_coefficients[1],
                                          ap2.array_coefficients[2]])
        expected_monomial_labels = [Multiset.from_list(l) for l in [[0], [1], [2], [3], [2, 2]]]
        expected_constant_term = ap1.constant_term + ap2.constant_term

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)


class TestArrayPolynomial(QiskitDynamicsTestCase):
    """Test the ArrayPolynomial class."""

    def setUp(self):
        """Set up typical polynomials including edge cases."""

        self.constant_0d = ArrayPolynomial(constant_term=3.0)
        self.constant_22d = ArrayPolynomial(constant_term=np.eye(2))
        self.non_constant_0d = ArrayPolynomial(
            array_coefficients=np.array([1.0, 2.0, 3.0]), monomial_labels=[[0], [1], [2]]
        )
        self.non_constant_32d = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 3, 2),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]),
        )
        self.non_constant_complex = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
        )

    def test_validation_error_no_ops(self):
        """Test validation error when no information specified."""
        with self.assertRaisesRegex(QiskitError, "At least one"):
            ArrayPolynomial()

    def test_trace_validation(self):
        """Test attempting to trace an AP with ndim < 2 raises an error."""
        with self.assertRaisesRegex(QiskitError, "at least 2."):
            self.non_constant_0d.trace()

    def test_only_constant_term(self):
        """Test construction and evaluation with only a constant term."""
        self.assertAllClose(self.constant_0d(), 3.0)

    def test_shape(self):
        """Test shape property."""
        self.assertTrue(self.constant_22d.shape == (2, 2))
        self.assertTrue(self.non_constant_0d.shape == tuple())
        self.assertTrue(self.non_constant_32d.shape == (3, 2))

    def test_ndim(self):
        """Test ndim."""
        self.assertTrue(self.constant_22d.ndim == 2)
        self.assertTrue(self.non_constant_0d.ndim == 0)
        self.assertTrue(self.non_constant_32d.ndim == 2)

    def test_transpose(self):
        """Test transpose."""
        trans = self.constant_0d.transpose()
        self.assertAllClose(trans.constant_term, 3.0)
        self.assertTrue(trans.array_coefficients is None)
        self.assertTrue(trans.monomial_labels == self.constant_0d.monomial_labels)

        trans = self.non_constant_32d.transpose()
        self.assertAllClose(trans.constant_term, self.non_constant_32d.constant_term.transpose())
        self.assertAllClose(
            trans.array_coefficients, self.non_constant_32d.array_coefficients.transpose((0, 2, 1))
        )
        self.assertTrue(trans.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_conj(self):
        """Test conj."""
        conj = self.constant_0d.conj()
        self.assertAllClose(conj.constant_term, 3.0)
        self.assertTrue(conj.array_coefficients is None)
        self.assertTrue(conj.monomial_labels == self.constant_0d.monomial_labels)

        conj = self.non_constant_complex.conj()
        self.assertAllClose(conj.constant_term, self.non_constant_complex.constant_term.conj())
        self.assertAllClose(
            conj.array_coefficients, self.non_constant_complex.array_coefficients.conj()
        )
        self.assertTrue(conj.monomial_labels == self.non_constant_complex.monomial_labels)

    def test_trace(self):
        """Test trace."""
        poly_trace = self.non_constant_32d.trace()

        self.assertAllClose(poly_trace.constant_term, self.non_constant_32d.constant_term.trace())
        self.assertAllClose(
            poly_trace.array_coefficients,
            self.non_constant_32d.array_coefficients.trace(axis1=1, axis2=2),
        )
        self.assertTrue(poly_trace.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_call_simple_case(self):
        """Typical expected usage case."""

        rng = np.random.default_rng(18471)
        coeffs = rng.uniform(low=-1, high=1, size=(5, 10, 10))
        monomial_labels = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]

        ap = ArrayPolynomial(coeffs, monomial_labels)

        c = np.array([3.0, 4.0])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

        c = np.array([3.2123, 4.1])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

    def test_compute_monomials_simple_case(self):
        """Simple test case for compute_monomials."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(938122)
        c = rng.uniform(size=(2,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [c[0], c[1], c[0] * c[0], c[0] * c[1], c[1] * c[1], c[0] * c[0] * c[0]]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_skipped_variable(self):
        """Test compute monomials case with skipped variable."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 2: 1}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(22321)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[2],
                c[0] * c[0],
                c[0] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_medium_case(self):
        """Test compute_monomials medium complexity test case."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]

        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_vectorized(self):
        """Test vectorized evaluation."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3, 20))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_only_first_order_terms(self):
        """Test a case with only first order terms."""

        multiset_list = [Multiset({0: 1}), Multiset({1: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(coeffs, multiset_list)

        c = np.array([3.0, 2.0])
        self.assertAllClose(ap.compute_monomials(c), c)

    def test_compute_monomials_incomplete_list(self):
        """Test case where the multiset_list is unordered and incomplete."""

        multiset_list = [Multiset({2: 2}), Multiset({0: 1}), Multiset({1: 1, 2: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(coeffs, multiset_list)

        c = np.array([3.0, 2.0, 4.0])
        self.assertAllClose(ap.compute_monomials(c), np.array([16.0, 3.0, 8.0]))


class TestArrayPolynomialJax(TestArrayPolynomial, TestJaxBase):
    """JAX version of TestArrayPolynomial."""

    def test_jit_compute_monomials(self):
        """Test jitting works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(coeffs, multiset_list)

        monomial_function_jit = jit(mp.compute_monomials)

        rng = np.random.default_rng(4122)
        c = rng.uniform(size=(2,))

        self.assertAllClose(mp.compute_monomials(c), monomial_function_jit(c))

    def test_compute_monomials_grad(self):
        """Test grad works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(coeffs, multiset_list)

        monomial_function_jit_grad = jit(grad(lambda c: mp.compute_monomials(c).sum()))

        c = np.array([2.0, 3.0])
        expected = np.array([1.0 + 0.0 + 4.0 + 3.0 + 0.0, 0.0 + 1.0 + 0.0 + 2.0 + 6.0])

        self.assertAllClose(expected, monomial_function_jit_grad(c))
