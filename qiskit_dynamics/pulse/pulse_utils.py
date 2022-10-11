# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
Pulse utils for computing dressed states and probabilities
"""


from typing import Optional, Union, List, Dict

import numpy as np
import numpy.linalg as la
from qiskit import QiskitError
from qiskit.quantum_info import DensityMatrix, Statevector


def tensor_state_label_generator(
    subsystem_dims: List[int], as_str: Optional[bool] = True
) -> List[Union[str, List[int]]]:
    """Generate ordered classical state labels for a tensor product system.

    Subsystem ordering in the labels is the reverse of the dimensions given in ``subsystem_dims``.
    E.g. with ``subsystem_dims = [3, 2]``, the returned labels are
    ``['00', '01', '02', '10', '11', '12']``. The ``as_str`` optional argument indicates whether
    to return as a list or as a string. This defaults to ``True``, but if ``False``, individual
    labels are returned in a list format, e.g. the string-formatted label ``'01'`` will be
    returned as the list ``[0, 1]``.

    Args:
        subsystem_dims: The dimensions of each subsystem.
        as_str: Return labels as string, defaults to True.

    Returns:
        List of system labels either in a string or list format.
    """
    # initialize labels list
    zero_prefactor = [0] * (len(subsystem_dims) - 1)
    labels = [zero_prefactor + [x] for x in range(subsystem_dims[0])]

    for idx, dim in enumerate(subsystem_dims[1:]):
        new_labels = []
        for counter in range(1, dim):
            for label in labels:
                new_label = label.copy()
                new_label[-(idx + 2)] = counter
                new_labels.append(new_label)
        labels += new_labels

    if as_str:
        labels = [[str(x) for x in lab] for lab in labels]
        labels = ["".join(lab) for lab in labels]

    return labels


def get_dressed_state_data(
    static_hamiltonian: np.ndarray, subsystem_dims: List[int]
) -> Union[Dict[str, np.ndarray], List[float], Dict[str, float]]:
    """Assuming it is nearly diagonal, get the eigenvalues and corresponding dressed states for
    a Hamiltonian.

    This function is essentially a wrapper around ``numpy.linalg.eigh``, but
    sorts the eigenvectors according to the value of ``np.argmax(np.abs(evec))``. It also
    validates that this is unique for each eigenvector.

    Args:
        static_hamiltonian: Static part of a Hamiltonian.
        subsystem_dims: Dimensions of the subsystems composing the system.
    Raises:
        QiskitError: If ``np.argmax(np.abs(evec))`` is non-unique across eigenvectors.
    Returns:
        Tuple: a pair of arrays, one containing eigenvalues and one containing corresponding
        eigenvectors.
    """

    evals, evecs = la.eigh(static_hamiltonian)

    dressed_evals = np.zeros_like(evals)
    dressed_states = np.zeros_like(evecs)

    found_positions = []
    for eval, evec in zip(evals, evecs):

        position = np.argmax(np.abs(evec))
        if position in found_positions:
            raise QiskitError("Dressed-state sorting failed due to overlap.")
        else:
            found_positions.append(position)

        dressed_states[position] = evec
        dressed_evals[position] = eval

    return dressed_evals, dressed_states


def compute_probabilities(
    state: Union[np.ndarray, list, Statevector, DensityMatrix], basis_states: dict
) -> Dict[str, float]:
    """Compute the probabilities for each state occupation using the formula for each basis state:
        For each basis state d, given input state vector s, we have the probability
       .. math::
        P(d) = (d^* \\cdot s)^2

        Or for a density matrix s
       .. math::
        P(d) = (s * (matmul) d).conj() * d

    Args:
        state: State vector for current state of system.
        basis_states: Dressed state dictionary for system.

    Raises:
        QiskitError: state vector or density matrix has too many dimensions
    Returns:
        Dict[str: float]: Dictionary of probabilities for each dressed state.
    """

    state = np.array(state)
    if state.ndim == 1:
        probs = {
            label: (np.abs(np.inner(basis_states[label].conj(), state) ** 2)).real
            for label in basis_states.keys()
        }
    elif state.ndim == 2:
        probs = {
            label: (
                np.abs(np.matmul(np.matmul(state, basis_states[label]).conj(), basis_states[label]))
            )
            for label in basis_states.keys()
        }

    else:
        raise QiskitError("State has too many dimensions")

    sum_probs = sum(list(probs.values()))
    probs = {key: value / sum_probs for key, value in probs.items()}

    return probs


def sample_counts(probs: Dict[str, float], n_shots: int, seed: Optional[int] = None) -> List[str]:
    """Sample the probability distribution `n_shot` times.

    Args:
        probs: Probability of dressed state occupation.
        n_shots: Number of samples
        seed: seed for random choice

    Returns:
        list of samples.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(list(probs.keys()), size=n_shots, p=list(probs.values()))
