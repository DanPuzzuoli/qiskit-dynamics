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

from qiskit import QiskitError
from qiskit.quantum_info import Statevector
import uuid
from random import sample
import datetime
import time
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import
from qiskit.result.models import ExperimentResult#, Result
import logging

from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel, ControlChannel

from qiskit_dynamics import Solver
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics.pulse.backend_parser.string_model_parser import parse_hamiltonian_dict

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule, ScheduleBlock#, block_to_schedule

from qiskit.transpiler import Target
from qiskit.result import Result

from qiskit_dynamics.pulse.pulse_utils import sample_counts, compute_probabilities, get_dressed_state_data

#Logger
logger = logging.getLogger(__name__)


class PulseSimulator(BackendV2):
    def __init__(
        self,
        solver: Solver,
        subsystem_dims,
        subsystem_labels: Optional[List[int]] = None,
        name: Optional[str] = 'PulseSimulator',
    ):
        """This init needs fleshing out. Need to determine all that is necessary for each use case.

        Assumptions
            - Solver is well-formed.
            - Solver Hamiltonian operators are specified in undressed basis using standard
              tensor-product convention (whether dense or sparse evaluation)

        Design questions
            - Simulating measurement requires subsystem dims and labels, should we allow this
              to be constructed without them, and then measurement is just not possible?
            - Should we add the ability to do custom measurements? I think yes, we do want
              things to be fully customizable.
        """
        # what to put for provider?
        super().__init__(
            provider=None,
            name=name,
            description='Pulse enabled simulator backend.',
            backend_version=0.1
        )
        self.solver = solver

        if subsystem_labels is None:
            subsystem_labels = np.arange(len(subsystem_dims), dtype=int)

        # get the static hamiltonian in the lab frame and undressed basis
        # assumes that the solver was constructed with operators specified in lab frame
        # using standard tensor product structure
        static_hamiltonian = None
        if isinstance(self.solver.model, HamiltonianModel):
            static_hamiltonian = self.solver.model.static_operator
        else:
            static_hamiltonian = self.solver.model.static_hamiltonian

        rotating_frame = self.solver.model.rotating_frame
        static_hamiltonian = 1j * rotating_frame.generator_out_of_frame(
            t=0., operator=-1j * static_hamiltonian
        )

        # get the dressed states
        dressed_evals, dressed_states = get_dressed_state_data(static_hamiltonian, subsystem_dims)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states

    @classmethod
    def from_backend(
        cls,
        backend: BackendV2,
        subsystem_list: Optional[List[int]] = None
    ) -> 'PulseSimulator':
        """Create a PulseSimulator object from a backend."""
        sim_backend = cls(solver=solver_from_backend(backend, subsystem_list))
        sim_backend.name = f'Pulse Simulator of {backend.name}'
        if isinstance(backend, BackendV1):
            raise QiskitError('from_backend not implemented for V1 backends.')
            #sim_backend.qubit_properties = backend.properties().qubit_property
            #sim_backend.target = Target()
            #sim_backend.target.qubit_properties = [sim_backend.qubit_properties(i) if i in subsystem_list else None for i in range(backend.configuration().n_qubits)]
        else:
            """
            Need to implement pruning of properties based on subsystem_list here.

            This may need to take data from sim_backend.
            """

            #sim_backend.qubit_properties = backend.qubit_properties
            #sim_backend.target = backend.target
            #sim_backend.drive_channel = backend.drive_channel
            #sim_backend.control_channel = backend.control_channel

            # sim_backend.target = Target()
            # sim_backend.target.qubit_properties = [sim_backend.qubit_properties(i) if i in subsystem_list else None for i in range(backend.configuration().n_qubits)]
            # sim_backend.target.dt = backend.dt
            sim_backend._meas_map = backend.meas_map
            #sim_backend.base_backend = backend
        return sim_backend

    def run(
        self,
        experiments: Union[QuantumCircuit, Schedule, ScheduleBlock],
        y0 = None,
        validate: Optional[bool] = False,
        solver_options: Optional[dict] = None,
        **options
    ) -> Result:
        """Run on the backend.

        Should return counts, for now just return probabilities
        """

        if validate:
            _validate_experiments(experiments)

        if y0 is None:
            y0 = Statevector(self._dressed_states[0])

        #job_id = str(uuid.uuid4())
        output = self._run(experiments, y0, job_id='', solver_options=solver_options)
        return output


    def _run(self, experiments, y0, job_id='', format_result=True, solver_options=None):
        """Run a job"""
        # Start timer
        start = time.time()

        if not isinstance(experiments, list):
            experiments = [experiments]

        if solver_options is None:
            solver_options = {}

        t_span = [[0, sched.duration * self.solver._dt] for sched in experiments]
        output = self.solver.solve(t_span=t_span, y0=y0, signals=experiments, **solver_options)
        return output

        #output = result_dict_from_sol(self.solver.solve(t_span = t_span, y0=y0, signals=experiments))

        ######
        # Do we need this output validation step? It seems like it might have more to do with
        # C++ interfacing from Aer
        ######


        # Validate output
        #if not isinstance(output, dict):
        #    logger.error("%s: simulation failed.", self.name)
        #    if output:
        #        logger.error('Output: %s', output)
        #    raise QiskitError(
        #        "simulation terminated without returning valid output.")


        # Display warning if simulation failed
        #if not output.get("success", False):
        #    msg = "Simulation failed"
        #    if "status" in output:
        #        msg += f" and returned the following error message:\n{output['status']}"
        #    logger.warning(msg)

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name
        output["backend_version"] = self.backend_version

        # Add execution time
        output["time_taken"] = time.time() - start

        if format_result:
            return format_results(output)
        return output

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            meas_map: The grouping of measurements which are multiplexed

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        # this needs to be implemented
        pass

    def acquire_channel(self, qubit: Iterable[int]) -> Union[int, AcquireChannel, None]:
        pass

    def drive_channel(self, qubit: int) -> Union[int, DriveChannel, None]:
        pass

    def control_channel(self, qubit: int) -> Union[int, ControlChannel, None]:
        pass

    def measure_channel(self, qubit: int) -> Union[int, MeasureChannel, None]:
        pass

    def _default_options(self):
        pass

    def max_circuits(self):
        pass

    def target(self):
        pass


def _validate_experiments(experiment, accept_list=True):
    """Raise errors if the experiment inputs are invalid.

    Notes:
        - not crazy about the "experiments" naming convention here
        - does this actually need to return anything, or just raise errors?
    """
    if isinstance(experiment, list) and accept_list:
        # if list apply recursively, but no longer accept lists
        for e in experiments:
            _validate_experiments(e, accept_list=False)
    elif not isinstance(experiment, (Schedule, ScheduleBlock)):
        raise QiskitError(f"Experiment type {type(experiment)} not supported by PulseSimulator.run.")


def get_counts(state_vector: np.ndarray, n_shots: int, seed: int) -> Dict[str, int]:
    """
    Get the counts from a state vector.
    """

    probs = compute_probabilities(state_vector, basis_states=convert_to_dressed(static_ham, subsystem_dims=subsystem_dims))
    counts = sample_counts(probs,n_shots=n_shots,seed=seed)
    return counts

def solver_from_backend(backend: Backend, subsystem_list: List[int]) -> 'PulseSimulator':
    """
    Create a solver object from a backend.
    """

    if isinstance(backend, BackendV2):
        ham_dict = backend.hamiltonian
    else:
        ham_dict = backend.configuration().hamiltonian

    static_hamiltonian, hamiltonian_operators, reduced_channels, subsystem_dims = parse_hamiltonian_dict(ham_dict, subsystem_list)

    drive_channels = [chan for chan in reduced_channels if 'd' in chan]
    control_channels = [chan for chan in reduced_channels if 'u' in chan]
    channel_freq_dict = {channel: freq for channel, freq in zip(reduced_channels, [freq for i,freq in enumerate(backend.defaults().qubit_freq_est) if i in subsystem_list])}

    for edge in backend.coupling_map.get_edges():
        channel_freq_dict[backend.control_channel(edge)[0].name] = backend.defaults().qubit_freq_est[edge[0]]


    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        rotating_frame=np.diag(static_hamiltonian),
        dt = backend.dt,
        hamiltonian_channels=reduced_channels,
        channel_carrier_freqs=channel_freq_dict
    )
    return solver

def result_dict_from_sol(sol: OdeResult, return_type: str = None) -> ExperimentResult:
    """
    Get the data from a solver object.
    """

    if return_type is None:
        if len(sol.y.shape) == 2:
            return_type = 'unitary'
        elif len(sol.y.shape) == 1:
            return_type = 'state_vector'
        else:
            raise NotImplementedError

    if return_type == 'state_vector':
        result_data = {'state_vector': sol.y}
    elif return_type == 'unitary':
        result_data = {'unitary': sol.y}
    elif return_type == 'counts':
        result_data = {'counts': get_counts(sol.y)}
    else:
        raise NotImplementedError(f"Return type {return_type} not implemented.")

    result_dict = {'results':[{'data': result_data, 'shots': 0, 'success': True}], 'success': True, 'qobj_id': ''}
    # result_dict['shots'] = 0
    # result_dict['success'] = True
    return result_dict

def format_results(output):
    """Format simulator output for constructing Result"""
    for result in output["results"]:
        data = result.get("data", {})
        metadata = result.get("metadata", {})
        save_types = metadata.get("result_types", {})
        save_subtypes = metadata.get("result_subtypes", {})
        for key, val in data.items():
            if key in save_types:
                data[key] = format_save_type(val, save_types[key], save_subtypes[key])
    return Result.from_dict(output)
