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

import numpy as np

from qiskit import pulse
from qiskit.pulse.transforms.canonicalization import block_to_schedule

from qiskit_dynamics.signals.signals import DiscreteSignal
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics.solvers.solver_classes import Solver

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
                 backend):
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

        @classmethod
        def from_Solver(cls,
                        solver,
                        hamiltonian_channels,
                        dissipator_channels,
                        carrier_freqs,
                        dt,
                        measurement_dict,
                        backend):
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
                               backend=backend)

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

    def solve(self, schedules, y0, **kwargs):
        """Output types are an issue for the JAX execution. `Solver.solve`
        will automatically take certain actions based on input type, and will wrap output states in
        the appropriate type, but we will need to unwrap this to be able to jit (which will introduce
        multiple pointless loops in which the outputs are wrapped, unwrapped, and then re-wrapped again).
        """
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
        schedules = new_schedules
        if self.backend != 'jax':
            results = []
            for sched in schedules:

                self.solver.signals = self.get_signals(sched)

                T = self.dt * sched.duration
                results.append(self.solver.solve(t_span=[0, T], y0=y0, **kwargs))

            return results
        else:

            converted_schedules = []
            durations = []
            max_duration = 0

            for sched in schedules:
                converted_schedules.append(self.get_signals(sched))
                durations.append(sched.duration)
                max_duration = max(max_duration, sched.duration)

            carrier_freqs = np.array([sig.carrier_freq for sig in converted_schedules[0]])


            def sim_function(sample_list, duration):
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

                return Array(solver_copy.solve(t_span=[0, duration * self.dt],
                                               y0=y0,
                                               wrap_results=False,
                                               **kwargs).y).data

            jit_sim_function = jit(sim_function)

            results = []
            zero_shape = (len(converted_schedules[0]), max_duration)
            for duration, signals in zip(durations, converted_schedules):
                sample_list = np.zeros(zero_shape, dtype=complex)
                for idx, sig in enumerate(signals):
                    sample_list[idx, 0:duration] = np.array(sig.samples)
                res = jit_sim_function(sample_list, duration)
                results.append(res)
                """ Need to rewrap in OdeResult objects here - whatever format is relevant
                """


            return results
