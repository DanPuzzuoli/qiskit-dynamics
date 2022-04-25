# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

r"""
Solver classes.
"""


from typing import Optional, Union, Tuple, Any, Type, List
from copy import copy

import numpy as np

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import pulse, QiskitError
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, Statevector, DensityMatrix

from qiskit_dynamics.models import (
    BaseGeneratorModel,
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals

from .solver_functions import solve_lmde
from .solver_utils import is_lindblad_model_vectorized, is_lindblad_model_not_vectorized


class NewSolver:

    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        hamiltonian_channels=None,
        dissipator_channels=None,
        carrier_freqs=None,
        dt=None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        validate: bool = True,
    ):
        """Initialize solver with model information.

        ******* NOTES:
        - Removes the ability to pass hamiltonian_signals and dissipator_signals to constructor,
            - Need to re-add w/deprecation warning
            - Problem: rwa_cutoff_freq requires signal frequencies to function, need to somehow
              add this back
        - Make sure simulating no signals or empty schedule works

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                                is specified, the ``frame_operator`` will be subtracted from
                                the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which
                            are diagonal can be supplied as a 1d array of the diagonal elements,
                            to explicitly indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized. See class documentation for a more
                            detailed explanation on how this argument affects object behaviour.
            evaluation_mode: Method for model evaluation. See documentation for
                             ``HamiltonianModel.evaluation_mode`` or
                             ``LindbladModel.evaluation_mode``.
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.
        """

        model = None
        if static_dissipators is None and dissipator_operators is None:
            model = HamiltonianModel(
                static_operator=static_hamiltonian,
                operators=hamiltonian_operators,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
        else:
            model = LindbladModel(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                static_dissipators=static_dissipators,
                dissipator_operators=dissipator_operators,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )

        """
        what to do about this?
        self._rwa_signal_map = None
        if rwa_cutoff_freq is not None:
            model, rwa_signal_map = rotating_wave_approximation(
                model, rwa_cutoff_freq, return_signal_map=True
            )
            self._rwa_signal_map = rwa_signal_map
        """
        self._model = model


        """
        Pulse stuff. Will probably need a bunch of validation
        """
        self.hamiltonian_channels = hamiltonian_channels
        self.dissipator_channels = dissipator_channels
        self.carrier_freqs = carrier_freqs

        self.all_channels = []
        if self.hamiltonian_channels is not None:
            for chan in self.hamiltonian_channels:
                if chan not in self.all_channels:
                    self.all_channels.append(chan)
        if self.dissipator_channels is not None:
            for chan in self.dissipator_channels:
                if chan not in self.all_channels:
                    self.all_channels.append(chan)

        self.all_channels.sort()
        self.dt = dt

        # need to be careful with this
        self.converter = None
        if dt is not None and carrier_freqs is not None and len(self.all_channels) > 0:
            self.converter = InstructionToSignals(dt=dt,
                                                  carriers=carrier_freqs,
                                                  channels=self.all_channels)

    @property
    def model(self) -> Union[HamiltonianModel, LindbladModel]:
        """The model of the system, either a Hamiltonian or Lindblad model."""
        return self._model

    def copy(self) -> "Solver":
        """Return a copy of self."""
        return copy(self)

    def solve(
        self,
        signals,
        t_span: Optional[Array] = None,
        y0: Optional[Union[Array, QuantumState, BaseOperator]] = None,
        dt_span: Optional[Array] = None,
        wrap_results=True,
        **kwargs
    ) -> OdeResult:
        r"""Solve the dynamical problem."""

        # need tons o' validation

        # determine whether to pass to
        if isinstance(signals, (Schedule, ScheduleBlock)) or (isinstance(signals, list) and isinstance(signals[0], (Schedule, ScheduleBlock))):
            schedules = to_schedule_list(signals)

            # assume converter defined here
            signals = []
            for schedule in schedules:
                sched_signals = self.converter.get_signals(schedule)

                if isinstance(self.model, HamiltonianModel):
                    signals.append([sched_signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels])
                else:
                    hamiltonian_signals = None
                    dissipator_signals = None

                    if self.hamiltonian_channels is not None:
                        hamiltonian_signals = [sched_signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels]
                    if self.dissipator_channels is not None:
                        dissipator_signals = [sched_signals[self.all_channels.index(chan)] for chan in self.dissipator_channels]

                    signals.append((hamiltonian_signals, dissipator_signals))

            """Setup simulation times - below is just a quick solution for demo,
            but need to validate that only one of dt_span and t_span is specified

            May also want to validate that dt_span is only being used when schedules
            are being simulated, as it may be a bit annoying to validate this as well.
            """

            if dt_span is None:
                dt_span = [[0, sched.duration] for sched in schedules]

            if t_span is None:
                # assume dt specified, can support both modes of operation
                t_span = Array(dt_span) * self.dt

        return self._solve_signals(signals, t_span, y0, wrap_results=wrap_results, **kwargs)

    def _solve_signals(
        self,
        signals,
        t_span: Optional[Array] = None,
        y0: Optional[Union[Array, QuantumState, BaseOperator]] = None,
        wrap_results=True,
        **kwargs
    ) -> OdeResult:
        r"""Solve the dynamical problem."""

        # convert types
        if isinstance(y0, QuantumState) and isinstance(self.model, LindbladModel):
            y0 = DensityMatrix(y0)

        y0, y0_cls = initial_state_converter(y0, return_class=True)

        # validate types
        if (y0_cls is SuperOp) and is_lindblad_model_not_vectorized(self.model):
            raise QiskitError(
                """Simulating SuperOp for a LinbladModel requires setting
                vectorized evaluation. Set LindbladModel.evaluation_mode to a vectorized option.
                """
            )

        # modify initial state for some custom handling of certain scenarios
        y_input = y0

        # if Simulating density matrix or SuperOp with a HamiltonianModel, simulate the unitary
        if y0_cls in [DensityMatrix, SuperOp] and isinstance(self.model, HamiltonianModel):
            y0 = np.eye(self.model.dim, dtype=complex)
        # if LindbladModel is vectorized and simulating a density matrix, flatten
        elif (
            (y0_cls is DensityMatrix)
            and isinstance(self.model, LindbladModel)
            and "vectorized" in self.model.evaluation_mode
        ):
            y0 = y0.flatten(order="F")

        # validate y0 shape before passing to solve_lmde
        if isinstance(self.model, HamiltonianModel) and (
            y0.shape[0] != self.model.dim or y0.ndim > 2
        ):
            raise QiskitError("""Shape mismatch for initial state y0 and HamiltonianModel.""")
        if is_lindblad_model_vectorized(self.model) and (
            y0.shape[0] != self.model.dim**2 or y0.ndim > 2
        ):
            raise QiskitError(
                """Shape mismatch for initial state y0 and LindbladModel
                                 in vectorized evaluation mode."""
            )
        if is_lindblad_model_not_vectorized(self.model) and y0.shape[-2:] != (
            self.model.dim,
            self.model.dim,
        ):
            raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel.""")

        """
        Need to handle cases:
            - signals is a list of signals (single Hamitonian simulation)
            - signals is a list of list of signals (multiple Hamiltonian simulation)
            - signals is a tuple of lists of signals (single Lindblad simulation)
            - signals is a list of tuples of lists of signals (multiple lindblad simulation)
        """

        num_signal_sets = 0
        if isinstance(signals, tuple):
            # should be single lindblad simulation
            # a tuple of lists of signals
            signals = [signals]
        elif isinstance(signals, list) and isinstance(signals[0], tuple):
            # should be multiple rounds of lindblad simulation
            # a list of tuples of lists of signals
            num_signal_sets = len(signals)
        elif isinstance(signals, list) and isinstance(signals[0], (list, SignalList)):
            # should be multiple rounds of Hamiltonian simulation
            # a list of lists or a list of SignalLists
            num_signal_sets = len(signals)
        elif isinstance(signals, SignalList) or (isinstance(signals, list) and isinstance(signals[0], Signal)):
            # should be single round of hamiltonian simulation
            # a list of signals or a SignalList
            signals = [signals]
        else:
            raise Exception("signals not understood")

        # setup t_span to have the same "shape" as signals
        t_span = Array(t_span)
        if t_span.ndim > 2:
            raise Exception("t_span must be either 1d or 2d")
        elif t_span.ndim == 2:
            if num_signal_sets == 0:
                raise Exception("t_span can only be a list of signals specifies multiple sets of signals")
            elif len(t_span) != num_signal_sets:
                raise Exception("t_span must specify the same number of simulations...")
        else:
            t_span = [t_span] * max(num_signal_sets, 1)

        all_results = []
        for _signals, _t_span in zip(signals, t_span):

            # need to add rwa handling
            self.model.signals = _signals

            results = solve_lmde(generator=self.model, t_span=_t_span, y0=y0, **kwargs)

            # handle special cases
            if y0_cls is DensityMatrix and isinstance(self.model, HamiltonianModel):
                # conjugate by unitary
                out = Array(results.y)
                results.y = out @ y_input @ out.conj().transpose((0, 2, 1))
            elif y0_cls is SuperOp and isinstance(self.model, HamiltonianModel):
                # convert to SuperOp and compose
                out = Array(results.y)
                results.y = (
                    np.einsum("nka,nlb->nklab", out.conj(), out).reshape(
                        out.shape[0], out.shape[1] ** 2, out.shape[1] ** 2
                    )
                    @ y_input
                )
            elif (y0_cls is DensityMatrix) and is_lindblad_model_vectorized(self.model):
                results.y = Array(results.y).reshape((len(results.y),) + y_input.shape, order="F")

            if y0_cls is not None and wrap_results:
                results.y = [final_state_converter(yi, y0_cls) for yi in results.y]

            all_results.append(results)

        # revert to empty model
        self.model.signals = None

        # strip the list wrapping
        if num_signal_sets == 0:
            all_results = all_results[0]

        return all_results


def initial_state_converter(
    obj: Any, return_class: bool = False
) -> Union[Array, Tuple[Array, Type]]:
    """Convert initial state object to an Array.

    Args:
        obj: An initial state.
        return_class: Optional. If True return the class to use
                      for converting the output y Array.

    Returns:
        Array: the converted initial state if ``return_class=False``.
        tuple: (Array, class) if ``return_class=True``.
    """
    # pylint: disable=invalid-name
    y0_cls = None
    if isinstance(obj, Array):
        y0, y0_cls = obj, None
    if isinstance(obj, QuantumState):
        y0, y0_cls = Array(obj.data), obj.__class__
    elif isinstance(obj, QuantumChannel):
        y0, y0_cls = Array(SuperOp(obj).data), SuperOp
    elif isinstance(obj, (BaseOperator, Gate, QuantumCircuit)):
        y0, y0_cls = Array(Operator(obj.data)), Operator
    else:
        y0, y0_cls = Array(obj), None
    if return_class:
        return y0, y0_cls
    return y0


def final_state_converter(obj: Any, cls: Optional[Type] = None) -> Any:
    """Convert final state Array to custom class. If ``cls`` is not ``None``,
    will explicitly convert ``obj`` into a ``numpy.array`` before wrapping,
    under the assumption that ``cls`` will be a ``qiskit.quantum_info`` type,
    which only support ``numpy.array``s.

    Args:
        obj: final state Array.
        cls: Optional. The class to convert to.

    Returns:
        Any: the final state.
    """
    if cls is None:
        return obj

    return cls(np.array(obj))

def to_schedule_list(schedules):
    if not isinstance(schedules, list):
        schedules = [schedules]

    new_schedules = []
    for sched in schedules:
        if isinstance(sched, pulse.ScheduleBlock):
            new_schedules.append(block_to_schedule(sched))
        elif isinstance(sched, pulse.Schedule):
            new_schedules.append(sched)
        else:
            raise Exception('invalid Schedule type')
    return new_schedules
