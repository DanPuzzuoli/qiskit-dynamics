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
# pylint: disable=invalid-name

"""
Pulse-enabled simulator backend.
"""

import datetime
import uuid

from typing import List, Optional, Union, Dict
import copy
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from qiskit import pulse
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj.common import QobjHeader
from qiskit.transpiler import Target
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers.options import Options
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.models.backendconfiguration import PulseBackendConfiguration
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qiskit import QiskitError, QuantumCircuit
from qiskit import schedule as build_schedule
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_dynamics.solvers.solver_classes import Solver

from .dynamics_job import DynamicsJob
from .backend_utils import (
    _get_dressed_state_decomposition,
    _get_lab_frame_static_hamiltonian,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
)
from .backend_string_parser import parse_backend_hamiltonian_dict


class DynamicsBackend(BackendV2):
    r"""Pulse enabled simulator backend.

    **Supported options**

    * ``shots``: Number of shots per experiment. Defaults to ``1024``.
    * ``solver``: The Qiskit Dynamics :class:`.Solver` instance used for simulation.
    * ``solver_options``: Dictionary containing optional kwargs for passing to :meth:`Solver.solve`,
      indicating solver methods and options. Defaults to the empty dictionary ``{}``.
    * ``subsystem_dims``: Dimensions of subsystems making up the system in ``solver``. Defaults to
      ``[solver.model.dim]``.
    * ``subsystem_labels``: Integer labels for subsystems. Defaults to ``[0, ...,
      len(subsystem_dims) - 1]``.
    * ``meas_map``: Measurement map. Defaults to ``[[idx] for idx in subsystem_labels]``.
    * ``initial_state``: Initial state for simulation, either the string ``"ground_state"``,
      indicating that the ground state for the system Hamiltonian should be used, or an arbitrary
      ``Statevector`` or ``DensityMatrix``. Defaults to ``"ground_state"``.
    * ``normalize_states``: Boolean indicating whether to normalize states before computing outcome
      probabilities. Defaults to ``True``. Setting to ``False`` can result in errors if the solution
      tolerance results in probabilities with significant numerical deviation from proper
      probability distributions.
    * ``meas_level``: Form of measurement return. Only supported value is ``2``, indicating that
      counts should be returned. Defaults to ``meas_level==2``.
    * ``max_outcome_level``: For ``meas_level==2``, the maximum outcome for each subsystem. Values
      will be rounded down to be no larger than ``max_outcome_level``. Must be a positive integer or
      ``None``. If ``None``, no rounding occurs. Defaults to ``1``.
    * ``memory``: Boolean indicating whether to return a list of explicit measurement outcomes for
      every experimental shot. Defaults to ``True``.
    * ``seed_simulator``: Seed to use in random sampling. Defaults to ``None``.
    * ``experiment_result_function``: Function for computing the ``ExperimentResult`` for each
      simulated experiment. This option defaults to :func:`default_experiment_result_function`, and
      any other function set to this option must have the same signature. Note that the default
      utilizes various other options that control results computation, and hence changing it will
      impact the meaning of other options.
    * ``configuration``: A :class:`PulseBackendConfiguration` instance or ``None``. This option
      defaults to ``None``, and is not required for the functioning of this class, but is provided
      for backwards compatibility. A set configuration will be returned by
      :meth:`DynamicsBackend.configuration()`.
    * ``defaults``: A :class:`PulseDefaults` instance or ``None``. This option
      defaults to ``None``, and is not required for the functioning of this class, but is provided
      for backwards compatibility. A set defaults will be returned by
      :meth:`DynamicsBackend.defaults()`.
    """

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        **options,
    ):
        """Instantiate with a :class:`.Solver` instance and additional options.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            options: Additional configuration options for the simulator.

        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """

        super().__init__(
            name="DynamicsBackend",
            description="Pulse enabled simulator backend.",
            backend_version="0.1",
        )

        # Dressed states of solver, will be calculated when solver option is set
        self._dressed_evals = None
        self._dressed_states = None
        self._dressed_states_adjoint = None

        # add subsystem_dims to options so set_options validation works
        if "subsystem_dims" not in options:
            options["subsystem_dims"] = [solver.model.dim]

        # Set simulator options
        self.set_options(solver=solver, **options)

        if self.options.subsystem_labels is None:
            labels = list(range(len(self.options.subsystem_dims)))
            self.set_options(subsystem_labels=labels)

        if self.options.meas_map is None:
            meas_map = [[idx] for idx in self.options.subsystem_labels]
            self.set_options(meas_map=meas_map)

        # self._target = target or Target() doesn't work as bool(target) can be False
        if target is None:
            target = Target()
        else:
            target = copy.copy(target)

        # add default simulator measure instructions
        instruction_schedule_map = target.instruction_schedule_map()
        for qubit in self.options.subsystem_labels:
            if not instruction_schedule_map.has(instruction="measure", qubits=qubit):
                with pulse.build() as meas_sched:
                    pulse.acquire(
                        duration=1, qubit_or_channel=qubit, register=pulse.MemorySlot(qubit)
                    )

            instruction_schedule_map.add(instruction="measure", qubits=qubit, schedule=meas_sched)

        self._target = target

    def _default_options(self):
        return Options(
            shots=1024,
            solver=None,
            solver_options={},
            subsystem_dims=None,
            subsystem_labels=None,
            meas_map=None,
            normalize_states=True,
            initial_state="ground_state",
            meas_level=MeasLevel.CLASSIFIED,
            max_outcome_level=1,
            memory=True,
            seed_simulator=None,
            experiment_result_function=default_experiment_result_function,
            configuration=None,
            defaults=None,
        )

    def set_options(self, **fields):
        """Set options for DynamicsBackend."""

        validate_subsystem_dims = False

        for key, value in fields.items():
            if not hasattr(self._options, key):
                raise AttributeError(f"Invalid option {key}")

            # validation checks
            if key == "initial_state":
                if value != "ground_state" and not isinstance(value, (Statevector, DensityMatrix)):
                    raise QiskitError(
                        """initial_state must be either "ground_state",
                        or a Statevector or DensityMatrix instance."""
                    )
            elif key == "meas_level" and value != 2:
                raise QiskitError("Only meas_level == 2 is supported by DynamicsBackend.")
            elif key == "max_outcome_level":
                if (value is not None) and (not isinstance(value, int) or (value <= 0)):
                    raise QiskitError("max_outcome_level must be a positive integer or None.")
            elif key == "experiment_result_function" and not callable(value):
                raise QiskitError("experiment_result_function must be callable.")
            elif key == "configuration" and not isinstance(value, PulseBackendConfiguration):
                raise QiskitError(
                    "configuration option must be an instance of PulseBackendConfiguration."
                )
            elif key == "defaults" and not isinstance(value, PulseDefaults):
                raise QiskitError("defaults option must be an instance of PulseDefaults.")

            # special setting routines
            if key == "solver":
                self._set_solver(value)
                validate_subsystem_dims = True
            else:
                if key == "subsystem_dims":
                    validate_subsystem_dims = True
                self._options.update_options(**{key: value})

        # perform additional consistency checks if certain options were modified
        if (
            validate_subsystem_dims
            and np.prod(self._options.subsystem_dims) != self._options.solver.model.dim
        ):
            raise QiskitError(
                "DynamicsBackend options subsystem_dims and solver.model.dim are inconsistent."
            )

    def _set_solver(self, solver):
        """Configure simulator based on provided solver."""
        if solver._dt is None:
            raise QiskitError(
                "Solver passed to DynamicsBackend is not configured for Pulse simulation."
            )

        self._options.update_options(solver=solver)
        # Get dressed states
        static_hamiltonian = _get_lab_frame_static_hamiltonian(solver.model)
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    # pylint: disable=arguments-differ
    def run(
        self,
        run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]],
        validate: Optional[bool] = True,
        **options,
    ) -> DynamicsJob:
        """Run a list of simulations.

        Args:
            run_input: A list of simulations, specified by ``QuantumCircuit``, ``Schedule``, or
                ``ScheduleBlock`` instances.
            validate: Whether or not to run validation checks on the input.
            **options: Additional run options to temporarily override current backend options.

        Returns:
            DynamicsJob object containing results and status.

        Raises:
            QiskitError: If invalid options are set.
        """

        if validate:
            _validate_run_input(run_input)

        # Configure run options for simulation
        if options:
            backend = copy.copy(self)
            backend.set_options(**options)
        else:
            backend = self

        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=backend)

        # get the acquires instructions and simulation times
        t_span, measurement_subsystems_list, memory_slot_indices_list = _get_acquire_data(
            schedules, backend.options.subsystem_labels
        )

        # Build and submit job
        job_id = str(uuid.uuid4())
        dynamics_job = DynamicsJob(
            backend=backend,
            job_id=job_id,
            fn=backend._run,
            fn_kwargs={
                "t_span": t_span,
                "schedules": schedules,
                "measurement_subsystems_list": measurement_subsystems_list,
                "memory_slot_indices_list": memory_slot_indices_list,
                "num_memory_slots_list": num_memory_slots_list,
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        """Simulate a list of schedules."""

        # simulate all schedules
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        # compute results for each experiment
        experiment_names = [schedule.name for schedule in schedules]
        rng = np.random.default_rng(self.options.seed_simulator)
        experiment_results = []
        for (
            experiment_name,
            solver_result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
        ) in zip(
            experiment_names,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        ):
            experiment_results.append(
                self.options.experiment_result_function(
                    experiment_name,
                    solver_result,
                    measurement_subsystems,
                    memory_slot_indices,
                    num_memory_slots,
                    self,
                    seed=rng.integers(low=0, high=9223372036854775807),
                )
            )

        # Construct full result object
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    @property
    def max_circuits(self):
        return None

    @property
    def target(self) -> Target:
        return self._target

    @property
    def meas_map(self) -> List[List[int]]:
        return self.options.meas_map

    def configuration(self) -> PulseBackendConfiguration:
        """Get the backend configuration."""
        return self.options.configuration

    def defaults(self) -> PulseDefaults:
        """Get the backend defaults."""
        return self.options.defaults

    @classmethod
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        subsystem_list: Optional[List[int]] = None,
        solver_init_kwargs: Optional[dict] = None,
        auto_rotating_frame: bool = True,
        **options,
    ) -> "DynamicsBackend":
        """Construct a :class:`.DynamicsBackend` instance from an existing ``Backend`` instance.

        The ``backend`` must have the ``configuration`` and ``defaults`` attributes. The
        ``configuration`` must containing a Hamiltonian description, step size ``dt``, number of
        qubits ``n_qubits``, and ``u_channel_lo`` (when control channels are present).  The
        ``defaults`` must contain ``qubit_freq_est`` and ``meas_freq_est``.

        The optional argument ``subsystem_list`` specifies which subset of qubits will be modelled
        in the constructed :class:`DynamicsBackend`, with all other qubits will being dropped from
        the model.

        Configuration of the underlying :class:`.Solver` is controlled via 
        ``solver_init_kwargs``, passed directly to :meth:`.Solver.__init__`. The additional
        argument ``auto_rotating_frame`` allows this method to automatically choose the rotating
        frame in which the :class:`.Solver` will be configured. If ``auto_rotating_frame==True``
        and no ``rotating_frame`` is specified in ``solver_init_kwargs``:

        * If a dense evaluation mode is chosen, the rotating frame will be set to the 
          ``static_hamiltonian`` indicated by the Hamiltonian in ``backend.configuration()``.
        * If a sparse evaluation mode is chosen, the rotating frame will be set to the diagonal of 
          ``static_hamiltonian``.
        
        **Technical notes**

        * The whole ``configuration`` and ``defaults`` attributes of the original backend will not 
          be copied into the constructed :class:`DynamicsBackend` instance, only the required data 
          stored within these attributes will be extracted. If required, for backwards 
          compatibility, ``'configuration'`` and ``'defaults'`` options can be set, which will be 
          returned via the :meth:`.configuration` and :meth:`.defaults` methods.
        * Gates and calibrations are not copied into the constructed :class:`DynamicsBackend`.
          Due to inevitable model inaccuracies, gates calibrated on a real device will not 
          have the same performance on the constructed :class:`DynamicsBackend`. As such, the 
          :class:`DynamicsBackend` will be constructed with an empty ``InstructionScheduleMap``, and
          must be recalibrated.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            solver_configuration: Additional keyword arguments to pass to the :class:`.Solver` instance
                constructed from the model in the backend.
            **options: Additional options to be applied in construction of the
                :class:`.DynamicsBackend`.

        Returns:
            DynamicsBackend

        Raises:
            QiskitError if any required parameters are missing from the passed backend.

        Notes:
            - Added configuration/defaults methods for "backwards compatibility". They are not
              strictly required by DynamicsBackend, but they are required of backends passed to
              DynamicsBackend.from_backend, so I think it's probably natural for cases utilizing
              from_backend for these to be present. These are settable as options, defaulting to
              None.
            - Configuration/defaults are not being copied over. They have way too many parameters
              that aren't relevant, and some which would need to be deleted (like default gates).
            - The new target is only being instantiated with dt.
            - 


        To do:
            - Issue with solver_kwargs is the user may will not have access to the hamiltonian
              terms, and hence can't effectively specify the rotating frame they want to be in. What
              to do about this? Could add string handling, like rotating_frame='static_hamiltonian'?
                - Can we somehow provide the user with "standard" configurations? E.g. if sparse, it
                  will automatically simulate in the diagonal of the static hamiltonian?
                - The user can set the rotating frame of the model AFTER construction as well (by
                  getting it from the solver).
                - Could maybe change arg to "solver_configuration", and it can either be a string
                  like "sparse", "dense", and the appropriate frame is automatically entered, or it
                  can be a dict and explicitly be passed as
            - To test:
                - all validation checks in from_backend, including option setting for
                  configuration/defaults


        """

        if not hasattr(backend, "configuration"):
            raise QiskitError(
                """DynamicsBackend.from_backend requires that the backend argument have a
                configuration attribute."""
            )
        if not hasattr(backend, "defaults"):
            raise QiskitError(
                """DynamicsBackend.from_backend requires that the backend argument have a defaults attribute."""
            )

        config = backend.configuration()
        defaults = backend.defaults()

        # get and parse Hamiltonian string dictionary
        if subsystem_list is not None:
            subsystem_list = sorted(subsystem_list)
            if subsystem_list[-1] >= config.n_qubits:
                raise QiskitError(
                    f"""subsystem_list contained {subsystem_list[-1]}, which is out of bounds for config.n_qubits == {config.n_qubits}."""
                )
        else:
            subsystem_list = list(range(config.n_qubits))

        if not hasattr(config, "hamiltonian"):
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has a hamiltonian attribute."
            )

        hamiltonian_dict = config.hamiltonian
        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            subsystem_dims,
        ) = parse_backend_hamiltonian_dict(hamiltonian_dict, subsystem_list)
        subsystem_dims = [subsystem_dims[idx] for idx in subsystem_list]

        # get time step size
        if not hasattr(config, "dt"):
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has a dt attribute."
            )
        dt = config.dt

        # construct model frequencies dictionary from backend
        channel_freqs = _get_backend_channel_freqs(
            backend_config=config, backend_defaults=defaults, channels=hamiltonian_channels
        )

        # build the solver
        solver_init_kwargs = copy.copy(solver_init_kwargs) or {}
        if auto_rotating_frame and "rotating_frame" not in solver_init_kwargs:
            evaluation_mode = solver_init_kwargs.get("evaluation_mode", "dense")
            if "dense" in evaluation_mode:
                solver_init_kwargs["rotating_frame"] = static_hamiltonian
            else:
                solver_init_kwargs["rotating_frame"] = np.diag(static_hamiltonian)

        solver = Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
            **solver_init_kwargs,
        )
        
        return cls(
            solver=solver,
            target=Target(dt=dt),
            subsystem_labels=subsystem_list,
            subsystem_dims=subsystem_dims,
            **options,
        )


def default_experiment_result_function(
    experiment_name: str,
    solver_result: OdeResult,
    measurement_subsystems: List[int],
    memory_slot_indices: List[int],
    num_memory_slots: Union[None, int],
    backend: DynamicsBackend,
    seed: Optional[int] = None,
) -> ExperimentResult:
    """Default routine for generating ExperimentResult object.

    Transforms state out of rotating frame into lab frame using ``backend.options.solver``,
    normalizes if ``backend.options.normalize_states==True``, and computes measurement results
    in the dressed basis based on measurement-related options in ``backend.options`` along with
    the measurement specification extracted from the experiments, passed as args to this function.

    Args:
        experiment_name: Name of experiment.
        solver_result: Result object from :class:`Solver.solve`.
        measurement_subsystems: Labels of subsystems in the model being measured.
        memory_slot_indices: Indices of memory slots to store the results in for each subsystem.
        num_memory_slots: Total number of memory slots in the returned output. If ``None``,
            ``max(memory_slot_indices)`` will be used.
        backend: The backend instance that ran the simulation. Various options and properties
            are utilized.
        seed: Seed for any random number generation involved (e.g. when computing outcome samples).

    Returns:
        ExperimentResult object containing results.

    Raises:
        QiskitError: If a specified option is unsupported.
    """

    yf = solver_result.y[-1]
    tf = solver_result.t[-1]

    # Take state out of frame, put in dressed basis, and normalize
    if isinstance(yf, Statevector):
        yf = np.array(backend.options.solver.model.rotating_frame.state_out_of_frame(t=tf, y=yf))
        yf = backend._dressed_states_adjoint @ yf
        yf = Statevector(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.linalg.norm(yf.data)
    elif isinstance(yf, DensityMatrix):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(t=tf, operator=yf)
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = DensityMatrix(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.diag(yf.data).sum()

    if backend.options.meas_level == MeasLevel.CLASSIFIED:

        # compute probabilities for measurement slot values
        measurement_subsystems = [
            backend.options.subsystem_labels.index(x) for x in measurement_subsystems
        ]
        memory_slot_probabilities = _get_memory_slot_probabilities(
            probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=backend.options.max_outcome_level,
        )

        # sample
        memory_samples = _sample_probability_dict(
            memory_slot_probabilities, shots=backend.options.shots, seed=seed
        )
        counts = _get_counts_from_samples(memory_samples)

        # construct results object
        exp_data = ExperimentResultData(
            counts=counts, memory=memory_samples if backend.options.memory else None
        )
        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.CLASSIFIED,
            seed=seed,
            header=QobjHeader(name=experiment_name),
        )
    else:
        raise QiskitError(f"meas_level=={backend.options.meas_level} not implemented.")


def _validate_run_input(run_input, accept_list=True):
    """Raise errors if the run_input is not one of QuantumCircuit, Schedule, ScheduleBlock, or
    a list of these.
    """
    if isinstance(run_input, list) and accept_list:
        # if list apply recursively, but no longer accept lists
        for x in run_input:
            _validate_run_input(x, accept_list=False)
    elif not isinstance(run_input, (QuantumCircuit, Schedule, ScheduleBlock)):
        raise QiskitError(f"Input type {type(run_input)} not supported by DynamicsBackend.run.")


def _get_acquire_data(schedules, valid_subsystem_labels):
    """Get the required data from the acquire commands in each schedule.

    Additionally validates that each schedule has acquire instructions occuring at one time,
    at least one memory slot is being listed, and all measured subsystems exist in
    subsystem_labels.
    """
    t_span_list = []
    measurement_subsystems_list = []
    memory_slot_indices_list = []
    for schedule in schedules:
        schedule_acquires = []
        schedule_acquire_times = []
        for start_time, inst in schedule.instructions:
            # only track acquires saving in a memory slot
            if isinstance(inst, pulse.Acquire) and inst.mem_slot is not None:
                schedule_acquires.append(inst)
                schedule_acquire_times.append(start_time)

        # validate
        if len(schedule_acquire_times) == 0:
            raise QiskitError(
                """At least one measurement saving a a result in a MemorySlot
                must be present in each schedule."""
            )

        for acquire_time in schedule_acquire_times[1:]:
            if acquire_time != schedule_acquire_times[0]:
                raise QiskitError("DynamicsBackend.run only supports measurements at one time.")

        t_span_list.append([0, schedule_acquire_times[0]])
        measurement_subsystems = []
        memory_slot_indices = []
        for inst in schedule_acquires:
            if inst.channel.index in valid_subsystem_labels:
                measurement_subsystems.append(inst.channel.index)
            else:
                raise QiskitError(
                    f"""Attempted to measure subsystem {inst.channel.index},
                    but it is not in subsystem_list."""
                )

            memory_slot_indices.append(inst.mem_slot.index)

        measurement_subsystems_list.append(measurement_subsystems)
        memory_slot_indices_list.append(memory_slot_indices)

    return t_span_list, measurement_subsystems_list, memory_slot_indices_list


def _to_schedule_list(
    run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]], backend: BackendV2
):
    """Convert all inputs to schedules, and store the number of classical registers present
    in any circuits.
    """
    if not isinstance(run_input, list):
        run_input = [run_input]

    schedules = []
    num_memslots = []
    for sched in run_input:
        num_memslots.append(None)
        if isinstance(sched, ScheduleBlock):
            schedules.append(block_to_schedule(sched))
        elif isinstance(sched, Schedule):
            schedules.append(sched)
        elif isinstance(sched, QuantumCircuit):
            num_memslots[-1] = sched.cregs[0].size
            schedules.append(build_schedule(sched, backend, dt=backend.options.solver._dt))
        else:
            raise QiskitError(f"Type {type(sched)} cannot be converted to Schedule.")
    return schedules, num_memslots


def _get_backend_channel_freqs(
    backend_config: PulseBackendConfiguration, backend_defaults: PulseDefaults, channels: List[str]
) -> Dict[str, float]:
    """Extract frequencies of channels from a backend configuration and defaults.

    Args:
        backend_config: A backend configuration object.
        backend_defaults: A backend defaults object.
        channels: Channel labels given as strings, assumed to be unique.

    Returns:
        Dict: Mapping of channel labels to frequencies.

    Raises:
        QiskitError: If the frequency for one of the channels cannot be found.
    """

    # partition types of channels
    drive_channels = []
    meas_channels = []
    u_channels = []

    for channel in channels:
        if channel[0] == "d":
            drive_channels.append(channel)
        elif channel[0] == "m":
            meas_channels.append(channel)
        elif channel[0] == "u":
            u_channels.append(channel)
        else:
            raise QiskitError("Unrecognized channel type requested.")

    # validate required attributes are present
    if drive_channels and not hasattr(backend_defaults, "qubit_freq_est"):
        raise QiskitError("DriveChannels in model but defaults does not have qubit_freq_est.")

    if meas_channels and not hasattr(backend_defaults, "meas_freq_est"):
        raise QiskitError("MeasureChannels in model but defaults does not have meas_freq_est.")

    if u_channels and not hasattr(backend_config, "u_channel_lo"):
        raise QiskitError("ControlChannels in model but configuration does not have u_channel_lo.")

    # populate frequencies
    channel_freqs = {}

    for channel in drive_channels:
        idx = int(channel[1:])
        if idx >= len(backend_defaults.qubit_freq_est):
            raise QiskitError(f"DriveChannel index {idx} is out of bounds.")
        channel_freqs[channel] = backend_defaults.qubit_freq_est[idx]

    for channel in meas_channels:
        idx = int(channel[1:])
        if idx >= len(backend_defaults.meas_freq_est):
            raise QiskitError(f"MeasureChannel index {idx} is out of bounds.")
        channel_freqs[channel] = backend_defaults.meas_freq_est[idx]

    for channel in u_channels:
        idx = int(channel[1:])
        if idx >= len(backend_config.u_channel_lo):
            raise QiskitError(f"ControlChannel index {idx} is out of bounds.")
        freq = 0.0
        for u_channel_lo in backend_config.u_channel_lo[idx]:
            freq += backend_defaults.qubit_freq_est[u_channel_lo.q] * u_channel_lo.scale

        channel_freqs[channel] = freq

    # validate that all channels have frequencies
    for channel in channels:
        if channel not in channel_freqs:
            raise QiskitError(f"No carrier frequency found for channel {channel}.")

    return channel_freqs
