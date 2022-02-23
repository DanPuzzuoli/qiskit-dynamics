# -*- coding: utf-8 -*-

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

from typing import Optional, Union, Tuple, Any, Type, List, Iterable
from copy import copy

import numpy as np

from scipy.integrate._ivp.ivp import OdeResult

from qiskit import pulse, QiskitError
from qiskit.pulse.transforms.canonicalization import block_to_schedule

from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import Statevector, DensityMatrix


from qiskit_dynamics.signals.signals import DiscreteSignal
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics.solvers.solver_classes import Solver, initial_state_converter

from qiskit_dynamics.array import Array

try:
    from jax import jit
except ImportError:
    pass


class PulseSolver:
    """Solver class for simulating pulse schedules."""

    def __init__(self,
                 static_hamiltonian,
                 hamiltonian_operators,
                 hamiltonian_channels,
                 static_dissipators,
                 dissipator_operators,
                 dissipator_channels,
                 carrier_freqs,
                 dt,
                 rotating_frame,
                 in_frame_basis,
                 evaluation_mode,
                 rwa_cutoff_freq,
                 validate,
                 backend,
                 subsystem_labels: Optional[List[int]] = None,
                 subsystem_dims: Optional[List[int]] = None):
        """NOTE: for now assuming that hamiltonian_channels and/or dissipator_channels
        have no internal repeats. We can add a step that merges operators with the
        same channel later.

        Do we want to allow this to work for constant models?
        """

        # first do some validation
        # will need to decide on how to handle defaults
        # e.g. for Hamiltonian operators:
        if hamiltonian_operators is not None:
            if hamiltonian_channels is None:
                raise Exception("gotta specify channels!")
            if len(hamiltonian_channels) != len(hamiltonian_operators):
                raise Exception("need same number of hamiltonian channels as operators!")

        # add validation for dissipators, etc.

        self.backend = backend or Array.default_backend()

        self.solver = Solver(static_hamiltonian=static_hamiltonian,
                             hamiltonian_operators=hamiltonian_operators,
                             static_dissipators=static_dissipators,
                             dissipator_operators=dissipator_operators,
                             rotating_frame=rotating_frame,
                             in_frame_basis=in_frame_basis,
                             evaluation_mode=evaluation_mode,
                             rwa_cutoff_freq=rwa_cutoff_freq,
                             validate=validate)

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
        self.converter = InstructionToSignals(dt=dt,
                                              carriers=carrier_freqs,
                                              channels=self.all_channels)

        ################################################################################################
        # Add dimension validation
        # Set labels to default to be the same lenght as dimensions
        ################################################################################################
        self.subsystem_labels = subsystem_labels or [0]
        self.subsystem_dims = subsystem_dims or [self.solver.model.dim]

    @classmethod
    def from_Solver(cls,
                    solver,
                    hamiltonian_channels,
                    dissipator_channels,
                    carrier_freqs,
                    dt,
                    measurement_dict,
                    backend,
                    subsystem_labels: Optional[List[int]] = None,
                    subsystem_dims: Optional[List[int]] = None):
        """Construct form an already-existing Solver instance."""

        return PulseSolver(static_hamiltonian=solver.static_hamiltonian,
                           hamiltonian_operators=solver.hamiltonian_operators,
                           hamiltonian_channels=hamiltonian_channels,
                           static_dissipators=solver.static_dissipators,
                           dissipator_operators=solver.dissipator_operators,
                           dissipator_channels=dissipator_channels,
                           carrier_freqs=carrier_freqs,
                           dt=dt,
                           rotating_frame=solver.rotating_frame,
                           in_frame_basis=solver.in_frame_basis,
                           evaluation_mode=solver.evaluation_mode,
                           rwa_cutoff_freq=solver.rwa_cutoff_freq,
                           validate=False,
                           backend=backend,
                           subsystem_labels=subsystem_labels,
                           subsystem_dims=subsystem_dims)

    def get_signals(self, schedule):
        """Get the signals from a schedule to pass to the Solver."""

        # currently the converter doesn't work if set to jax mode
        default_backend = Array.default_backend()
        Array.set_default_backend('numpy')

        signals = self.converter.get_signals(schedule)

        Array.set_default_backend(default_backend)

        if isinstance(self.solver.model, HamiltonianModel):
            return [signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels]

        hamiltonian_signals = None
        dissipator_signals = None

        if self.hamiltonian_channels is not None:
            hamiltonian_signals = [signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels]
        if self.dissipator_channels is not None:
            dissipator_signals = [signals[self.all_channels.index(chan)] for chan in self.dissipator_channels]

        return hamiltonian_signals, dissipator_signals

    def solve(self, schedules, y0, sample_span=None, wrap_results=True, **kwargs):
        """Output types are an issue for the JAX execution. `Solver.solve`
        will automatically take certain actions based on input type, and will wrap output states in
        the appropriate type, but we will need to unwrap this to be able to jit (which will introduce
        multiple pointless loops in which the outputs are wrapped, unwrapped, and then re-wrapped again).

        sample_span simulation time window behaviour:
            - If None defaults to simulating for the length of each schedule according to
              when the last pulse shape ends
            - If a single [t0, tf] pair, all schedules simulated over that interval
            - If a list of pairs [t0, tf], must be the same length as the schedules, and
              each schedule is simulated according to that pair
            - Note: specified as samples
        """

        schedules = to_schedule_list(schedules)

        # handle different sample_span arguments
        if sample_span is None:
            sample_span = [[0, sched.duration] for sched in schedules]
        elif isinstance(sample_span, Iterable):
            sample_span = list(sample_span)
            if isinstance(sample_span[0], int):
                sample_span = sample_span * len(schedules)
            elif len(sample_span) != len(schedules):
                raise QiskitError('if specifying a list of sample_span windows, must be same length as schedules')

        if self.backend != 'jax':
            results = []
            for sched, sim_range in zip(schedules, sample_span):

                self.solver.signals = self.get_signals(sched)

                t0 = self.dt * sim_range[0]
                tf = self.dt * sim_range[1]
                results.append(self.solver.solve(t_span=[t0, tf], y0=y0, wrap_results=wrap_results, **kwargs))

            return results
        else:

            converted_schedules = []
            max_duration = 0

            for sched in schedules:
                converted_schedules.append(self.get_signals(sched))
                max_duration = max(max_duration, sched.duration)

            carrier_freqs = np.array([sig.carrier_freq for sig in converted_schedules[0]])

            # determine output type
            if isinstance(y0, QuantumState) and isinstance(self.model, LindbladModel):
                y0 = DensityMatrix(y0)

            y0, y0_cls = initial_state_converter(y0, return_class=True)
            if y0_cls is not None:
                y0 = y0_cls(y0)

            def sim_function(sample_list, t0, tf):
                solver_copy = self.solver.copy()

                signals = [DiscreteSignal(dt=self.dt,
                                          samples=sample_list[k],
                                          carrier_freq=carrier_freqs[k])
                          for k in range(len(sample_list))]

                if isinstance(self.solver.model, HamiltonianModel) and self.hamiltonian_channels is not None:
                    solver_copy.signals = [signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels]
                else:
                    ham_signals = None
                    diss_signals = None
                    if self.hamiltonian_channels is not None:
                        ham_signals = [signals[self.all_channels.index(chan)] for chan in self.hamiltonian_channels]
                    if self.dissipator_signals is not None:
                        diss_signals = [signals[self.all_channels.index(chan)] for chan in self.dissipator_channels]

                    solver_copy.signals = (ham_signals, diss_signals)

                results = solver_copy.solve(t_span=[t0, tf],
                                               y0=y0,
                                               wrap_results=False,
                                               **kwargs)

                return Array(results.t).data, Array(results.y).data

            jit_sim_function = jit(sim_function)

            results = []
            zero_shape = (len(converted_schedules[0]), max_duration)
            for sim_range, signals in zip(sample_span, converted_schedules):
                sample_list = np.zeros(zero_shape, dtype=complex)
                for idx, sig in enumerate(signals):
                    sample_list[idx, 0:len(sig.samples)] = np.array(sig.samples)
                results_t, results_y = jit_sim_function(sample_list, sim_range[0] * self.dt, sim_range[1] * self.dt)

                # wrap results if desired
                if y0_cls is not None and wrap_results:
                    results_y = [y0_cls(y) for y in results_y]

                results.append(OdeResult(t=results_t, y=Array(results_y, backend="jax", dtype=complex)))


            return results

    def measurement_probabilities(self,
                                  y: Union[np.ndarray, QuantumState],
                                  subsystems_to_keep: Optional[List[int]] = None) -> np.ndarray:
        """Compute measurement probabilities.

        For now assumes things ordered in the standard basis.

        subsystems_to_keep indicates which subsystems to include in the output.
        """
        # this is a hack to get things into the right dimensions
        y = np.array(y)

        if y.ndim == 1:
            y = Statevector(y, dims=self.subsystem_dims)
        elif y.ndim == 2:
            y = DensityMatrix(y, dims=self.subsystem_dims)
        else:
            raise QiskitError('y is not a valid state.')

        full_probabilities = y.probabilities_dict()

        if subsystems_to_keep is None:
            return full_probabilities

        if any(label not in self.subsystem_labels for label in subsystems_to_keep):
            raise QiskitError('label not valid')

        reversed_subsystems = list(reversed(self.subsystem_labels))
        subsystem_indices = [reversed_subsystems.index(label) for label in subsystems_to_keep]

        # loop through and add up probabilities corresponding to the same state
        reduced_probabilities = dict()
        for state_label, prob in full_probabilities.items():
            reduced_label = ''.join(state_label[idx] for idx in subsystem_indices)
            if reduced_label in reduced_probabilities:
                reduced_probabilities[reduced_label] += prob
            else:
                reduced_probabilities[reduced_label] = prob

        return reduced_probabilities

    def run(self, schedules, y0, **kwargs):
        """Meant to mimic backend.run, includes measurement."""

        ################################################################################################
        # validate y0
        ################################################################################################
        schedules = to_schedule_list(schedules)

        # get the acquires instructions
        sim_spans = []
        measurement_subsystems = []
        for schedule in schedules:
            schedule_acquires = []
            schedule_acquire_times = []
            for start_time, inst in schedule.instructions:
                if isinstance(inst, pulse.Acquire):
                    schedule_acquires.append(inst)
                    schedule_acquire_times.append(start_time)

            # maybe need to validate more here
            validate_acquires(schedule_acquire_times, schedule_acquires)

            sim_spans.append([0, schedule_acquire_times[0]])
            measurement_subsystems.append([inst.channel.index for inst in schedule_acquires])
            measurement_subsystems[-1].sort()

            ############################################################################################
            # handle mem/reg slots here?
            ############################################################################################

        results = self.solve(schedules, y0, sample_span=sim_spans, **kwargs)

        for result, meas_subsystems in zip(results, measurement_subsystems):
            result.measurement_probabilities = self.measurement_probabilities(result.y[-1], subsystems_to_keep=meas_subsystems)

        return results



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


def validate_acquires(schedule_acquire_times, schedule_acquires):
    """Validate the acquire instructions.

    For now, make sure all acquires happen at oen time.
    """

    start_time = schedule_acquire_times[0]
    for time in schedule_acquire_times[1:]:
        if time != start_time:
            raise QiskitError("PulseSolver.run only supports measurements at one time.")
