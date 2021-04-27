# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Tests for signals.
"""

import numpy as np

from qiskit_ode.signals import Signal, Constant, DiscreteSignal
from qiskit_ode.signals.signals import SignalSum, DiscreteSignalSum
from qiskit_ode.dispatch import Array

from ..common import QiskitOdeTestCase, TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
except ImportError:
    pass


class TestSignal(QiskitOdeTestCase):
    """Tests for Signal object."""

    def setUp(self):
        self.signal1 = Signal(lambda t: 0.25, carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t ** 2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t ** 2) + 1j * t, carrier_freq=0.1, phase=-0.1)

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.signal1.envelope(0.0), 0.25)
        self.assertAllClose(self.signal1.envelope(1.23), 0.25)

        self.assertAllClose(self.signal2.envelope(1.1), 2 * (1.1**2))
        self.assertAllClose(self.signal2.envelope(1.23), 2 * (1.23**2))

        self.assertAllClose(self.signal3.envelope(1.1), 2 * (1.1**2) + 1j * 1.1)
        self.assertAllClose(self.signal3.envelope(1.23), 2 * (1.23**2) + 1j * 1.23)

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.signal1.envelope(t_vals), np.array([0.25, 0.25]))
        self.assertAllClose(self.signal2.envelope(t_vals), np.array([2 * (1.1**2), 2 * (1.23**2)]))
        self.assertAllClose(self.signal3.envelope(t_vals), np.array([2 * (1.1**2) + 1j * 1.1, 2 * (1.23**2) + 1j * 1.23]))

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.signal1.envelope(t_vals), np.array([[0.25, 0.25], [0.25, 0.25]]))
        self.assertAllClose(self.signal2.envelope(t_vals), np.array([[2 * (1.1**2), 2 * (1.23**2)],
                                                                     [2 * (0.1**2), 2 * (0.24**2)]]))
        self.assertAllClose(self.signal3.envelope(t_vals), np.array([[2 * (1.1**2) + 1j * 1.1, 2 * (1.23**2) + 1j * 1.23],
                                                                     [2 * (0.1**2) + 1j * 0.1, 2 * (0.24**2) + 1j * 0.24]]))

    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(self.signal1.complex_value(0.0), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0))
        self.assertAllClose(self.signal1.complex_value(1.23), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23))

        self.assertAllClose(self.signal2.complex_value(1.1), 2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1))
        self.assertAllClose(self.signal2.complex_value(1.23), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23))

        self.assertAllClose(self.signal3.complex_value(1.1), (2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)))
        self.assertAllClose(self.signal3.complex_value(1.23), (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)))

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.signal1.complex_value(t_vals), np.array([0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)]))
        self.assertAllClose(self.signal2.complex_value(t_vals), np.array([2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)]))
        self.assertAllClose(self.signal3.complex_value(t_vals), np.array([(2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)), (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))]))

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.signal1.complex_value(t_vals), np.array([[0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)],
                                                                          [0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.24)]]))
        self.assertAllClose(self.signal2.complex_value(t_vals), np.array([[2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)],
                                                                          [2 * (0.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.1), 2 * (0.24**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.24)]]))
        self.assertAllClose(self.signal3.complex_value(t_vals), np.array([[(2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)), (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))],
                                                                          [(2 * (0.1**2) + 1j * 0.1) * np.exp(1j * 2 * np.pi * 0.1 * 0.1 + 1j * (-0.1)), (2 * (0.24**2) + 1j * 0.24) * np.exp(1j * 2 * np.pi * 0.1 * 0.24 + 1j * (-0.1))]]))

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.signal1(0.0), np.real(0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0)))
        self.assertAllClose(self.signal1(1.23), np.real(0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)))

        self.assertAllClose(self.signal2(1.1), np.real(2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1)))
        self.assertAllClose(self.signal2(1.23), np.real(2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)))

        self.assertAllClose(self.signal3(1.1), np.real((2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1))))
        self.assertAllClose(self.signal3(1.23), np.real((2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))))

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.signal1(t_vals), np.array([0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)]).real)
        self.assertAllClose(self.signal2(t_vals), np.array([2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)]).real)
        self.assertAllClose(self.signal3(t_vals), np.array([(2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)), (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))]).real)

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.signal1(t_vals), np.array([[0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)],
                                                                          [0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.1), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.24)]]).real)
        self.assertAllClose(self.signal2(t_vals), np.array([[2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)],
                                                                          [2 * (0.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.1), 2 * (0.24**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.24)]]).real)
        self.assertAllClose(self.signal3(t_vals), np.array([[(2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)), (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))],
                                                                          [(2 * (0.1**2) + 1j * 0.1) * np.exp(1j * 2 * np.pi * 0.1 * 0.1 + 1j * (-0.1)), (2 * (0.24**2) + 1j * 0.24) * np.exp(1j * 2 * np.pi * 0.1 * 0.24 + 1j * (-0.1))]]).real)

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        sig3_conj = self.signal3.conjugate()

        self.assertAllClose(self.signal3.phase, -sig3_conj.phase)
        self.assertAllClose(self.signal3.carrier_freq, -sig3_conj.carrier_freq)
        self.assertAllClose(self.signal3.complex_value(1.231), np.conjugate(sig3_conj.complex_value(1.231)))
        self.assertAllClose(self.signal3.complex_value(1.231 * np.pi), np.conjugate(sig3_conj.complex_value(1.231 * np.pi)))


class TestConstant(QiskitOdeTestCase):
    """Tests for Constant object."""

    def setUp(self):
        self.constant1 = Constant(1.)
        self.constant2 = Constant(3. + 1j * 2)

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.constant1.envelope(0.0), 1.)
        self.assertAllClose(self.constant1.envelope(1.23), 1.)

        self.assertAllClose(self.constant2.envelope(1.1), 3.)
        self.assertAllClose(self.constant2.envelope(1.23), 3.)

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1.envelope(t_vals), np.array([1., 1.]))
        self.assertAllClose(self.constant2.envelope(t_vals), np.array([3., 3.]))

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.constant1.envelope(t_vals), np.array([[1., 1.], [1., 1.]]))
        self.assertAllClose(self.constant2.envelope(t_vals), np.array([[3., 3.], [3., 3.]]))


    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(self.constant1.complex_value(0.0), 1.)
        self.assertAllClose(self.constant1.complex_value(1.23), 1.)

        self.assertAllClose(self.constant2.complex_value(1.1), 3.)
        self.assertAllClose(self.constant2.complex_value(1.23), 3.)

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1.complex_value(t_vals), np.array([1., 1.]))
        self.assertAllClose(self.constant2.complex_value(t_vals), np.array([3., 3.]))

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.constant1.complex_value(t_vals), np.array([[1., 1.], [1., 1.]]))
        self.assertAllClose(self.constant2.complex_value(t_vals), np.array([[3., 3.], [3., 3.]]))

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.constant1(0.0), 1.)
        self.assertAllClose(self.constant1(1.23), 1.)

        self.assertAllClose(self.constant2(1.1), 3.)
        self.assertAllClose(self.constant2(1.23), 3.)

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1(t_vals), np.array([1., 1.]))
        self.assertAllClose(self.constant2(t_vals), np.array([3., 3.]))

        t_vals = np.array([[1.1, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.constant1(t_vals), np.array([[1., 1.], [1., 1.]]))
        self.assertAllClose(self.constant2(t_vals), np.array([[3., 3.], [3., 3.]]))

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        const_conj = self.constant2.conjugate()
        self.assertAllClose(const_conj(1.1), 3.)


class TestDiscreteSignal(QiskitOdeTestCase):
    """Tests for DiscreteSignal object."""

    def setUp(self):
        self.discrete1 = DiscreteSignal(dt=0.5, samples=np.array([1., 2., 3.]), carrier_freq=3.)
        self.discrete2 = DiscreteSignal(dt=0.5, samples=np.array([1. + 2j, 2. + 1j, 3.]), carrier_freq=1., phase=3.)

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.discrete1.envelope(0.0), 1.)
        self.assertAllClose(self.discrete1.envelope(1.23), 3.)

        self.assertAllClose(self.discrete2.envelope(0.1), 1. + 2j)
        self.assertAllClose(self.discrete2.envelope(1.23), 3.)

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([0.1, 1.23])
        self.assertAllClose(self.discrete1.envelope(t_vals), np.array([1., 3.]))
        self.assertAllClose(self.discrete2.envelope(t_vals), np.array([1. + 2j, 3.]))

        t_vals = np.array([[0.8, 1.23],
                           [0.1, 0.24]])
        self.assertAllClose(self.discrete1.envelope(t_vals), np.array([[2., 3.], [1., 1.]]))
        self.assertAllClose(self.discrete2.envelope(t_vals), np.array([[2. + 1j, 3.], [1 + 2j, 1. + 2j]]))

    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(self.discrete1.complex_value(0.0), 1.)
        self.assertAllClose(self.discrete1.complex_value(1.23), 3. * np.exp(1j * 2 * np.pi * 3.0 * 1.23))

        self.assertAllClose(self.discrete2.complex_value(0.1), (1. + 2j) * np.exp(1j * 2 * np.pi * 1.0 * 0.1 + 1j * 3.))
        self.assertAllClose(self.discrete2.complex_value(1.23), 3. * np.exp(1j * 2 * np.pi * 1.0 * 1.23 + 1j * 3.))

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([0.1, 1.23])
        phases = np.exp(1j * 2 * np.pi * 3. * t_vals)
        self.assertAllClose(self.discrete1.complex_value(t_vals), np.array([1., 3.]) * phases)
        phases = np.exp(1j * 2 * np.pi * 1. * t_vals + 1j * 3.)
        self.assertAllClose(self.discrete2.complex_value(t_vals), np.array([1. + 2j, 3.]) * phases)

        t_vals = np.array([[0.8, 1.23],
                           [0.1, 0.24]])
        phases = np.exp(1j * 2 * np.pi * 3. * t_vals)
        self.assertAllClose(self.discrete1.complex_value(t_vals), np.array([[2., 3.], [1., 1.]]) * phases)
        phases = np.exp(1j * 2 * np.pi * 1. * t_vals + 1j * 3.)
        self.assertAllClose(self.discrete2.complex_value(t_vals), np.array([[2. + 1j, 3.], [1 + 2j, 1. + 2j]]) * phases)

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.discrete1(0.0), 1.)
        self.assertAllClose(self.discrete1(1.23), np.real(3. * np.exp(1j * 2 * np.pi * 3.0 * 1.23)))

        self.assertAllClose(self.discrete2(0.1), np.real((1. + 2j) * np.exp(1j * 2 * np.pi * 1.0 * 0.1 + 1j * 3.)))
        self.assertAllClose(self.discrete2(1.23), np.real(3. * np.exp(1j * 2 * np.pi * 1.0 * 1.23 + 1j * 3.)))

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([0.1, 1.23])
        phases = np.exp(1j * 2 * np.pi * 3. * t_vals)
        self.assertAllClose(self.discrete1(t_vals), np.real(np.array([1., 3.]) * phases))
        phases = np.exp(1j * 2 * np.pi * 1. * t_vals + 1j * 3.)
        self.assertAllClose(self.discrete2(t_vals), np.real(np.array([1. + 2j, 3.]) * phases))

        t_vals = np.array([[0.8, 1.23],
                           [0.1, 0.24]])
        phases = np.exp(1j * 2 * np.pi * 3. * t_vals)
        self.assertAllClose(self.discrete1(t_vals), np.real(np.array([[2., 3.], [1., 1.]]) * phases))
        phases = np.exp(1j * 2 * np.pi * 1. * t_vals + 1j * 3.)
        self.assertAllClose(self.discrete2(t_vals), np.real(np.array([[2. + 1j, 3.], [1 + 2j, 1. + 2j]]) * phases))

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""
        discrete_conj = self.discrete2.conjugate()
        self.assertAllClose(discrete_conj.samples, np.conjugate(self.discrete2.samples))
        self.assertAllClose(discrete_conj.carrier_freq, -self.discrete2.carrier_freq)
        self.assertAllClose(discrete_conj.phase, -self.discrete2.phase)
        self.assertAllClose(discrete_conj.dt, self.discrete2.dt)


class TestSignalSum(QiskitOdeTestCase):
    """Test evaluation functions for ``SignalSum``."""

    def setUp(self):
        self.signal1 = Signal(np.vectorize(lambda t: 0.25), carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t ** 2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t ** 2) + 1j * t, carrier_freq=0.1, phase=-0.1)

        self.sig_sum1 = self.signal1 + self.signal2
        self.sig_sum2 = self.signal2 - self.signal3

    def test_envelope(self):
        """Test envelope evaluation."""
        t = 0.0
        self.assertAllClose(self.sig_sum1.envelope(t), [self.signal1.envelope(t), self.signal2.envelope(t)])
        self.assertAllClose(self.sig_sum2.envelope(t), [self.signal2.envelope(t), -self.signal3.envelope(t)])
        t = 1.23
        self.assertAllClose(self.sig_sum1.envelope(t), [self.signal1.envelope(t), self.signal2.envelope(t)])
        self.assertAllClose(self.sig_sum2.envelope(t), [self.signal2.envelope(t), -self.signal3.envelope(t)])

    def test_envelope_vectorized(self):
        """Test vectorized envelope evaluation."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(self.sig_sum1.envelope(t_vals), [[self.signal1.envelope(t), self.signal2.envelope(t)] for t in t_vals])
        self.assertAllClose(self.sig_sum2.envelope(t_vals), [[self.signal2.envelope(t), -self.signal3.envelope(t)] for t in t_vals])
        t_vals = np.array([[0.0, 1.23], [0.1, 2.]])
        self.assertAllClose(self.sig_sum1.envelope(t_vals), [[[self.signal1.envelope(t), self.signal2.envelope(t)] for t in t_row] for t_row in t_vals])
        self.assertAllClose(self.sig_sum2.envelope(t_vals), [[[self.signal2.envelope(t), -self.signal3.envelope(t)] for t in t_row] for t_row in t_vals])

    def test_complex_value(self):
        """Test complex_value evaluation."""
        t = 0.0
        self.assertAllClose(self.sig_sum1.complex_value(t), self.signal1.complex_value(t) + self.signal2.complex_value(t))
        self.assertAllClose(self.sig_sum2.complex_value(t), self.signal2.complex_value(t) - self.signal3.complex_value(t))
        t = 1.23
        self.assertAllClose(self.sig_sum1.complex_value(t), self.signal1.complex_value(t) + self.signal2.complex_value(t))
        self.assertAllClose(self.sig_sum2.complex_value(t), self.signal2.complex_value(t) - self.signal3.complex_value(t))

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(self.sig_sum1.complex_value(t_vals), [self.signal1.complex_value(t) + self.signal2.complex_value(t) for t in t_vals])
        self.assertAllClose(self.sig_sum2.complex_value(t_vals), [self.signal2.complex_value(t) - self.signal3.complex_value(t) for t in t_vals])
        t_vals = np.array([[0.0, 1.23], [0.1, 2.]])
        self.assertAllClose(self.sig_sum1.complex_value(t_vals), [[self.signal1.complex_value(t) + self.signal2.complex_value(t) for t in t_row] for t_row in t_vals])
        self.assertAllClose(self.sig_sum2.complex_value(t_vals), [[self.signal2.complex_value(t) - self.signal3.complex_value(t) for t in t_row] for t_row in t_vals])

    def test_call(self):
        """Test __call__."""
        t = 0.0
        self.assertAllClose(self.sig_sum1(t), self.signal1(t) + self.signal2(t))
        self.assertAllClose(self.sig_sum2(t), self.signal2(t) - self.signal3(t))
        t = 1.23
        self.assertAllClose(self.sig_sum1(t), self.signal1(t) + self.signal2(t))
        self.assertAllClose(self.sig_sum2(t), self.signal2(t) - self.signal3(t))

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(self.sig_sum1(t_vals), [self.signal1(t) + self.signal2(t) for t in t_vals])
        self.assertAllClose(self.sig_sum2(t_vals), [self.signal2(t) - self.signal3(t) for t in t_vals])
        t_vals = np.array([[0.0, 1.23], [0.1, 2.]])
        self.assertAllClose(self.sig_sum1(t_vals), [[self.signal1(t) + self.signal2(t) for t in t_row] for t_row in t_vals])
        self.assertAllClose(self.sig_sum2(t_vals), [[self.signal2(t) - self.signal3(t) for t in t_row] for t_row in t_vals])

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        sig_sum1_conj = self.sig_sum1.conjugate()
        self.assertAllClose(sig_sum1_conj.complex_value(2.313), np.conjugate(self.sig_sum1.complex_value(2.313)))
        self.assertAllClose(sig_sum1_conj.complex_value(0.1232), np.conjugate(self.sig_sum1.complex_value(0.1232)))


class TestDiscreteSignalSum(TestSignalSum):
    """Tests for DiscreteSignalSum."""

    def setUp(self):
        self.signal1 = Signal(np.vectorize(lambda t: 0.25), carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t ** 2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t ** 2) + 1j * t, carrier_freq=0.1, phase=-0.1)

        self.sig_sum1 = DiscreteSignalSum.from_SignalSum(self.signal1 + self.signal2, dt=0.5, start_time=0, n_samples=10)
        self.sig_sum2 = DiscreteSignalSum.from_SignalSum(self.signal2 - self.signal3, dt=0.5, start_time=0, n_samples=10)

        self.signal1 = DiscreteSignal.from_Signal(self.signal1, dt=0.5, start_time=0, n_samples=10)
        self.signal2 = DiscreteSignal.from_Signal(self.signal2, dt=0.5, start_time=0, n_samples=10)
        self.signal3 = DiscreteSignal.from_Signal(self.signal3, dt=0.5, start_time=0, n_samples=10)


class TestSignalJax(TestSignal, TestJaxBase):
    """Jax version of TestSignal."""


class TestConstantJax(TestSignal, TestJaxBase):
    """Jax version of TestConstant."""


class TestDiscreteSignalJax(TestDiscreteSignal, TestJaxBase):
    """Jax version of TestDiscreteSignal."""


class TestSignalSumJax(TestSignalSum, TestJaxBase):
    """Jax version of TestSignalSum."""


class TestDiscreteSignalSumJax(TestDiscreteSignalSum, TestJaxBase):
    """Jax version of TestSignalSum."""


class TestSignalsJaxTransformations(QiskitOdeTestCase, TestJaxBase):
    """Test cases for jax transformations of signals."""

    def setUp(self):
        self.signal = Signal(lambda t: t**2, carrier_freq=3.)
        self.constant = Constant(3 * np.pi)
        self.discrete_signal = DiscreteSignal(dt=0.5, samples=jnp.ones(20, dtype=complex), carrier_freq=2.)
        self.signal_sum = self.signal + self.discrete_signal
        self.discrete_signal_sum = DiscreteSignalSum.from_SignalSum(self.signal_sum, dt=0.5, n_samples=20)

    def test_jit_eval(self):
        """Test jit-compilation of signal evaluation."""
        self._test_jit_signal_eval(self.signal, t=2.1)
        self._test_jit_signal_eval(self.constant, t=2.1)
        self._test_jit_signal_eval(self.discrete_signal, t=2.1)
        self._test_jit_signal_eval(self.signal_sum, t=2.1)
        self._test_jit_signal_eval(self.discrete_signal_sum, t=2.1)

    def test_jit_grad_eval(self):
        """Test taking the gradient and then jitting signal evaluation functions."""
        t = 2.1
        self._test_grad_eval(self.signal,
                             t=t,
                             sig_deriv_val=2 * t * np.cos(2 * np.pi * 3. * t) + (t**2) * (- 2 * np.pi * 3) * np.sin(2 * np.pi * 3. * t),
                             complex_deriv_val=2 * t * np.exp(1j * 2 * np.pi * 3. * t) + (t**2) * (1j * 2 * np.pi * 3.) * np.exp(1j * 2 * np.pi * 3. * t))
        self._test_grad_eval(self.constant,
                             t=t,
                             sig_deriv_val=0.,
                             complex_deriv_val=0.)
        self._test_grad_eval(self.discrete_signal,
                             t=t,
                             sig_deriv_val=np.real(self.discrete_signal.samples[5]) * (-2 * np.pi * 2.) * np.sin(2 * np.pi * 2. * t),
                             complex_deriv_val=self.discrete_signal.samples[5] * (1j * 2 * np.pi * 2.) * np.exp(1j * 2 * np.pi * 2. * t))
        self._test_grad_eval(self.signal_sum,
                             t=t,
                             sig_deriv_val=2 * t * np.cos(2 * np.pi * 3. * t) + (t**2) * (- 2 * np.pi * 3) * np.sin(2 * np.pi * 3. * t) + np.real(self.discrete_signal.samples[5]) * (-2 * np.pi * 2.) * np.sin(2 * np.pi * 2. * t),
                             complex_deriv_val=2 * t * np.exp(1j * 2 * np.pi * 3. * t) + (t**2) * (1j * 2 * np.pi * 3.) * np.exp(1j * 2 * np.pi * 3. * t) + self.discrete_signal.samples[5] * (1j * 2 * np.pi * 2.) * np.exp(1j * 2 * np.pi * 2. * t))
        self._test_grad_eval(self.discrete_signal_sum,
                             t=t,
                             sig_deriv_val=(2.25 ** 2) * (- 2 * np.pi * 3) * np.sin(2 * np.pi * 3. * t) + np.real(self.discrete_signal.samples[5]) * (-2 * np.pi * 2.) * np.sin(2 * np.pi * 2. * t),
                             complex_deriv_val=(2.25 ** 2) * (1j * 2 * np.pi * 3.) * np.exp(1j * 2 * np.pi * 3. * t) + self.discrete_signal.samples[5] * (1j * 2 * np.pi * 2.) * np.exp(1j * 2 * np.pi * 2. * t))

    def _test_jit_signal_eval(self, signal, t=2.1):
        """jit compilation and evaluation of main signal functions."""
        sig_call_jit = jit(lambda t: Array(signal(t)).data)
        self.assertAllClose(sig_call_jit(t), signal(t))
        sig_envelope_jit = jit(lambda t: Array(signal.envelope(t)).data)
        self.assertAllClose(sig_envelope_jit(t), signal.envelope(t))
        sig_complex_value_jit = jit(lambda t: Array(signal.complex_value(t)).data)
        self.assertAllClose(sig_complex_value_jit(t), signal.complex_value(t))

    def _test_grad_eval(self, signal, t, sig_deriv_val, complex_deriv_val):
        """Test chained grad and jit compilation."""
        sig_call_jit = jit(grad(lambda t: Array(signal(t)).data))
        self.assertAllClose(sig_call_jit(t), sig_deriv_val)
        sig_complex_value_jit_re = jit(grad(lambda t: np.real(Array(signal.complex_value(t))).data))
        sig_complex_value_jit_imag = jit(grad(lambda t: np.imag(Array(signal.complex_value(t))).data))
        self.assertAllClose(sig_complex_value_jit_re(t), np.real(complex_deriv_val))
        self.assertAllClose(sig_complex_value_jit_imag(t), np.imag(complex_deriv_val))


class TestSignalsOLD(QiskitOdeTestCase):
    """Tests for signals."""

    def setUp(self):
        pass

    def test_constant(self):
        """Test Constant signal"""

        constant = Constant(0.5)

        self.assertAllClose(constant.envelope(0.0), 0.5)
        self.assertAllClose(constant.envelope(10.0), 0.5)
        self.assertAllClose(constant(0.0), 0.5)
        self.assertAllClose(constant(10.0), 0.5)

    def test_signal(self):
        """Test Signal."""

        # Signal with constant amplitude
        signal = Signal(0.25, carrier_freq=0.3)
        self.assertAllClose(signal.envelope(0.0), 0.25)
        self.assertAllClose(signal.envelope(1.23), 0.25)
        self.assertAllClose(signal(0.0), 0.25)
        self.assertAllClose(signal(1.0), 0.25 * np.cos(0.3 * 2.0 * np.pi))

        signal = Signal(0.25, carrier_freq=0.3, phase=0.5)
        self.assertAllClose(signal(1.0), 0.25 * np.cos(0.3 * 2.0 * np.pi + 0.5))

        # Signal with parabolic amplitude
        signal = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertAllClose(signal.envelope(0.0), 0.0)
        self.assertAllClose(signal.envelope(3.0), 18.0)
        self.assertAllClose(signal(0.0), 0.0)
        self.assertAllClose(signal(2.0), 8.0 * np.cos(0.1 * 2.0 * np.pi * 2.0))

        signal = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1, phase=-0.1)
        self.assertAllClose(signal(2.0), 8.0 * np.cos(0.1 * 2.0 * np.pi * 2.0 - 0.1))

    def test_piecewise_constant(self):
        """Test PWC signal."""

        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        piecewise_const = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        self.assertAllClose(piecewise_const.envelope(0.0), 0.0)
        self.assertAllClose(piecewise_const.envelope(2.0), 1.0)
        self.assertAllClose(piecewise_const(0.0), 0.0)
        self.assertAllClose(piecewise_const(3.0), 2.0 * np.cos(0.5 * 2.0 * np.pi * 3.0))

        piecewise_const = DiscreteSignal(
            dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5
        )
        self.assertAllClose(piecewise_const(3.0), 2.0 * np.cos(0.5 * 2.0 * np.pi * 3.0 + 0.5))

    def test_multiplication(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1 * const2, Signal))
        self.assertAllClose((const1 * const2)(0.0), 0.15)
        self.assertAllClose((const1 * const2)(10.0), 0.15)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 * signal1, Signal))
        self.assertTrue(isinstance(signal1 * const1, Signal))
        self.assertTrue(isinstance(signal1 * signal2, Signal))
        self.assertAllClose((signal1 * signal2).carrier_freq, Array([0.2, 0.0]))
        self.assertAllClose((signal1 * const1).carrier_freq, 0.1)
        self.assertAllClose((signal1 * signal2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose(
            (signal1 * signal2).envelope(3.0), 0.5 * Array([3.0 * 18.0, 3.0 * 18.0])
        )
        self.assertAllClose((signal1 * signal2)(0.0), 0.0)
        self.assertAllClose((signal1 * signal2)(2.0), signal1(2.0) * signal2(2.0))

        # Test piecewise constant
        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        pwc1 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 2.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.1
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(len(const1 * pwc1) == 1)
        self.assertTrue(isinstance((const1 * pwc1)[0], Signal))
        self.assertTrue(isinstance(signal1 * pwc1, SignalSum))
        self.assertTrue(isinstance(pwc1 * pwc2, SignalSum))
        self.assertTrue(isinstance((pwc1 * pwc2)[0], Signal))
        self.assertTrue(isinstance((pwc1 * pwc2)[1], Signal))
        self.assertTrue(len(pwc1 * const1) == 1)
        self.assertTrue(isinstance((pwc1 * const1)[0], DiscreteSignal))
        self.assertTrue(isinstance(pwc1 * signal1, SignalSum))
        self.assertTrue(isinstance((pwc1 * signal1)[0], Signal))

        # Test values
        self.assertAllClose((pwc1 * pwc2).carrier_freq, Array([0.5 + 0.1, 0.5 - 0.1]))
        self.assertAllClose((pwc1 * pwc2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose((pwc1 * pwc2).envelope(4.0), 0.5 * Array([1.0, 1.0]))
        self.assertAllClose((pwc1 * pwc2)(0.0), 0.0)
        self.assertAllClose((pwc1 * pwc2)(4.0), 1.0 * np.cos(0.6 * 2.0 * np.pi * 4.0))

        # Test phase
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5)
        self.assertAllClose((pwc1 * pwc2)(4.0), 1.0 * np.cos(0.6 * 2.0 * np.pi * 4.0 + 0.5))

    def test_addition(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1 + const2, Signal))
        self.assertAllClose((const1 + const2)(0.0), 0.8)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0 * t ** 2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 + signal1, Signal))
        self.assertTrue(isinstance(signal1 + const1, Signal))
        self.assertTrue(isinstance(signal1 + signal2, Signal))
        self.assertAllClose((signal1 + signal2).carrier_freq, Array([0.1, 0.1]))
        self.assertAllClose((signal1 + const1).carrier_freq, Array([0.1, 0.0]))
        self.assertAllClose((signal1 + signal2).envelope(0.0), Array([3.0, 0.0]))
        expected = Array([3.0, 2.0 * (3.0) ** 2])
        self.assertAllClose((signal1 + signal2).envelope(3.0), expected)
        self.assertAllClose((signal1 + signal2)(0.0), 3.0)
        self.assertAllClose((signal1 + signal2)(2.0), 11.0 * np.cos(0.1 * 2.0 * np.pi * 2.0))

        # Test piecewise constant
        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.5
        pwc1 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 1.0
        samples = Array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        carrier_freq = 0.1
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(isinstance(const1 + pwc1, Signal))
        self.assertTrue(isinstance(signal1 + pwc1, Signal))
        self.assertTrue(isinstance(pwc1 + pwc2, Signal))
        self.assertTrue(isinstance(pwc1 + const1, Signal))
        self.assertTrue(isinstance(pwc1 + signal1, Signal))

        # Test values
        self.assertAllClose((pwc1 + pwc2).carrier_freq, Array([0.5, 0.1]))

        self.assertAllClose((pwc1 + pwc2).envelope(0.0), Array([0.0, 0.0]))
        self.assertAllClose((pwc1 + pwc2).envelope(4.0), Array([1.0, 1.0]))
        self.assertAllClose((pwc1 + pwc2)(0.0), 0.0)
        expected = 1.0 * np.cos(0.5 * 2.0 * np.pi * 4.0) + 1.0 * np.cos(0.1 * 2.0 * np.pi * 4.0)
        self.assertAlmostEqual((pwc1 + pwc2)(4.0), expected, places=8)

        # Test phase
        pwc2 = DiscreteSignal(dt=dt, samples=samples, carrier_freq=carrier_freq, phase=0.5)
        self.assertAllClose((pwc1 + pwc2).envelope(4.0), Array([1.0, 1.0]))
