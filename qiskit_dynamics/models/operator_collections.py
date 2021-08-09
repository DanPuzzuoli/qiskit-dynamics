##COPYRIGHT STUFF

"""Generic operator for general linear maps"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from copy import deepcopy
import numpy as np
from scipy.sparse.csr import csr_matrix
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array, vec_commutator, vec_dissipator


class BaseOperatorCollection(ABC):
    r"""BaseOperatorCollection is an abstract class
    intended to store a general set of linear mappings {\Lambda_i}
    in order to implement differential equations of the form
    \dot{y} = \Lambda(y,t). Generically, \Lambda will be a sum of
    other linear maps \Lambda_i(y,t), which are in turn some
    combination of left-multiplication \Lambda_i(y,t) = A(t)y(t),
    right-multiplication \Lambda_i(y,t) = y(t)B(t), and both, with
    \Lambda_i(y,t) = A(t)y(t)B(t), but this implementation
    will only engage with these at the level of \Lambda(y,t)

    Drift is a property that represents some time-independent
    component \Lambda_d of the decpmoosition, which will be
    used to facilitate rotating frame transformations."""

    @property
    def drift(self) -> Array:
        """Returns drift part of operator collection."""
        return self._drift

    @drift.setter
    def drift(self, new_drift: Optional[Array] = None):
        """Sets Drift operator, if used."""
        self._drift = new_drift

    @property
    @abstractmethod
    def num_operators(self) -> int:
        """Returns number of operators the collection
        is storing."""
        pass

    @abstractmethod
    def evaluate_generator(self, signal_values: Array) -> Array:
        r"""If the model can be represented simply and
        without reference to the state involved, e.g.
        in the case \dot{y} = G(t)y(t) being represented
        as G(t), returns this independent representation.
        If the model cannot be represented in such a
        manner (c.f. Lindblad model), then errors."""
        pass

    @abstractmethod
    def evaluate_rhs(self, signal_values: Union[List[Array], Array], y: Array) -> Array:
        """Evaluates the model for a given state
        y provided the values of each signal
        component s_j(t). Must be defined for all
        models."""
        pass

    def __call__(
        self, signal_values: Union[List[Array], Array], y: Optional[Array] = None
    ) -> Array:
        """Evaluates the model given the values of the signal
        terms s_j, suppressing the choice between
        evaluate_rhs and evaluate_generator
        from the user. May error if y is not provided and
        model cannot be expressed without choice of state.
        """

        if y is None:
            return self.evaluate_generator(signal_values)

        return self.evaluate_rhs(signal_values, y)

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)


class DenseOperatorCollection(BaseOperatorCollection):
    r"""Meant to be a calculation object for models that
    only ever need left multiplication–those of the form
    \dot{y} = G(t)y(t), where G(t) = \sum_j s_j(t) G_j + G_d.
    Can evaluate G(t) independently of y.
    """

    @property
    def num_operators(self) -> int:
        return self._operators.shape[-3]

    def evaluate_generator(self, signal_values: Array) -> Array:
        r"""Evaluates the operator G at time t given
        the signal values s_j(t) as G(t) = \sum_j s_j(t)G_j"""
        if self._drift is None:
            return np.tensordot(signal_values, self._operators, axes=1)
        else:
            return np.tensordot(signal_values, self._operators, axes=1) + self._drift

    def evaluate_rhs(self, signal_values: Array, y: Array) -> Array:
        """Evaluates the product G(t)y"""
        return np.dot(self.evaluate_generator(signal_values), y)

    def __init__(self, operators: Array, drift: Optional[Array] = None):
        """Initialize
        Args:
            operators: (k,n,n) Array specifying the terms G_j
            drift: (n,n) Array specifying the extra drift G_d
        """
        self._operators = to_array(operators)
        self.drift = drift


class SparseOperatorCollection(BaseOperatorCollection):
    r"""Meant to be a calculation object for models that
    only ever need left multiplication–those of the form
    \dot{y} = G(t)y(t), where G(t) = \sum_j s_j(t) G_j + G_d.
    Can evaluate G(t) independently of y. G_j and G_d are
    sparse matrices.
    """

    @property
    def num_operators(self) -> int:
        return self._operators.shape[0]

    @property
    def drift(self) -> Array:
        return super().drift

    @drift.setter
    def drift(self, new_drift):
        if isinstance(new_drift, csr_matrix):
            self._drift = new_drift
        else:
            self._drift = csr_matrix(new_drift)

    def __init__(
        self,
        operators: Array,
        drift: Optional[Array] = None,
        tol: Optional[int] = 10,
    ):
        """Initialize
        Args:
            operators: (k,n,n) Array specifying the terms G_j
            drift: (n,n) Array specifying the drift term G_d
            tol: Values will be rounded at tol places after decimal place.
                Chosen to avoid storing excess sparse entries for entries
                sufficiently close to zero."""
        self.drift = np.round(drift, tol)
        self._operators = np.empty(shape=operators.shape[0], dtype="O")
        for i in range(operators.shape[0]):
            self._operators[i] = csr_matrix(np.round(operators[i], tol))

    def evaluate_generator(self, signal_values: Array) -> csr_matrix:
        r"""Evaluates the operator G at time t given
        the signal values s_j(t) as G(t) = \sum_j s_j(t)G_j
        Returns:
            Generator as sparse array"""
        signal_values = signal_values.reshape(1, signal_values.shape[-1])
        if self._drift is None:
            return np.tensordot(signal_values, self._operators, axes=1)[0]
        else:
            return np.tensordot(signal_values, self._operators, axes=1)[0] + self._drift

    def evaluate_rhs(self, signal_values: Array, y: Array) -> Array:
        if len(y.shape) == 2:
            # For y a matrix with y[:,i] storing the i^{th} state, it is faster to
            # first evaluate the generator in most cases
            gen = np.tensordot(signal_values, self._operators, axes=1) + self.drift
            return gen.dot(y)
        elif len(y.shape) == 1:
            # for y a vector, it is typically faster to use the following, very
            # strange-looking implementation
            tmparr = np.empty(shape=(1), dtype="O")
            tmparr[0] = y
            return np.dot(signal_values, self._operators * tmparr) + self.drift.dot(y)


class DenseLindbladCollection(BaseOperatorCollection):
    r"""Intended to be the calculation object for the Lindblad equation
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t)
        (L_j\rho L_j^\dagger - (1/2) * {L_j^\daggerL_j,\rho})
    where [,] and {,} are the operator commutator and anticommutator, respectively.
    In the case that the Hamiltonian is also a function of time, varying as
    H(t) = H_d + \sum_j s_j(t) H_j, where H_d is the drift term,
    this can be further decomposed. We will allow for both our dissipator terms
    and our Hamiltonian terms to have different signal decompositions.
    """

    @property
    def dissipator_operators(self):
        """Gets dissipator operators as a (k,n,n) Array"""
        return self._dissipator_operators

    @dissipator_operators.setter
    def dissipator_operators(self, dissipator_operators: Array):
        self._dissipator_operators = dissipator_operators
        if dissipator_operators is not None:
            self._dissipator_operators_conj = np.conjugate(
                np.transpose(dissipator_operators, [0, 2, 1])
            ).copy()
            self._dissipator_products = np.matmul(
                self._dissipator_operators_conj, self._dissipator_operators
            )

    @property
    def num_operators(self):
        return self._hamiltonian_operators.shape[-3], self._dissipator_operators.shape[-3]

    def __init__(
        self,
        hamiltonian_operators: Array,
        drift: Array,
        dissipator_operators: Optional[Array] = None,
    ):
        r"""Converts an array of Hamiltonian components and signals,
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            drift: If supplied, treated as a constant term to be added to the
                Hamiltonian of the system.
            dissipator_operators: the terms L_j in Lindblad equation.
                (m,n,n) array.
        """

        self._hamiltonian_operators = hamiltonian_operators
        self.dissipator_operators = dissipator_operators
        self.drift = drift

    def evaluate_generator(self, signal_values: Array) -> Array:
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_hamiltonian(self, signal_values: Array) -> Array:
        r"""Gets the Hamiltonian matrix, as calculated by the model,
        and used for the commutator -i[H,y]
        Args:
            signal_values: [Real] values of s_j in H = \sum_j s_j(t) H_j
        Returns:
            Hamiltonian matrix."""
        return np.tensordot(signal_values, self._hamiltonian_operators, axes=1) + self.drift

    def evaluate_rhs(self, signal_values: List[Array], y: Array) -> Array:
        r"""Evaluates Lindblad equation RHS given a pair of signal values
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation as (A+B)y + y(A-B) + C, where
            A = (-1/2)*\sum_j\gamma(t) L_j^\dagger L_j
            B = -iH
            C = \sum_j \gamma_j(t) L_j y L_j^\dagger
        Args:
            signal_values: length-2 list of Arrays. has the following components
                signal_values[0]: hamiltonian signal values, s_j(t)
                    Must have length self._num_ham_terms
                signal_values[1]: dissipator signal values, \gamma_j(t)
                    Must have length self._num_dis_terms
            y: density matrix as (n,n) Array representing the state at time t
        Returns:
            RHS of Lindblad equation -i[H,y] + \sum_j\gamma_j(t)
            (L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y})
        """

        hamiltonian_matrix = -1j * self.evaluate_hamiltonian(signal_values[0])  # B matrix

        if self._dissipator_operators is not None:
            dissipators_matrix = (-1 / 2) * np.tensordot(  # A matrix
                signal_values[1], self._dissipator_products, axes=1
            )

            left_mult_contribution = np.matmul(hamiltonian_matrix + dissipators_matrix, y)
            right_mult_contribution = np.matmul(y, -hamiltonian_matrix + dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works
                y = np.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )

            both_mult_contribution = np.tensordot(
                signal_values[1],
                np.matmul(
                    self._dissipator_operators, np.matmul(y, self._dissipator_operators_conj)
                ),
                axes=(-1, -3),
            )  # C

            return left_mult_contribution + right_mult_contribution + both_mult_contribution

        else:
            return np.dot(hamiltonian_matrix, y) - np.dot(y, hamiltonian_matrix)


class DenseVectorizedLindbladCollection(DenseOperatorCollection):
    r"""Intended as a calculation object for the Lindblad equation,
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t)
            (L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y})
    where all left-, right-, and left-and-right multiplication is
    handled by vectorization, a process by which \rho, an (n,n)
    matrix, is embedded in a vector space of dimension n^2 using
    the column stacking convention, in which the matrix [a,b;c,d]
    is written as [a,c,b,d]."""

    def __init__(
        self,
        hamiltonian_operators: Array,
        drift: Array,
        dissipator_operators: Optional[Array] = None,
    ):
        r"""Converts an array of Hamiltonian components and signals,
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation using only left-multiplication.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            drift: Constant term to be added to the Hamiltonian of the system.
            dissipator_operators: the terms L_j in Lindblad equation.
                (m,n,n) array.
        """

        # Convert Hamiltonian to commutator formalism
        self._hamiltonian_operators = hamiltonian_operators
        self._drift_terms = drift
        vec_drift = -1j * vec_commutator(drift)

        vec_ham_ops = -1j * vec_commutator(to_array(hamiltonian_operators))
        total_ops = None
        if dissipator_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(dissipator_operators))
            total_ops = np.append(vec_ham_ops, vec_diss_ops, axis=0)
        else:
            total_ops = vec_ham_ops

        super().__init__(total_ops, drift=vec_drift)

    def evaluate_hamiltonian(self, signal_values: Array) -> Array:
        r"""Gets the Hamiltonian matrix, as calculated by the model,
        and used for the commutator -i[H,y]
        Args:
            signal_values: [Real] values of s_j in H = \sum_j s_j(t) H_j
        Returns:
            Hamiltonian matrix."""
        return (
            np.tensordot(signal_values, self._hamiltonian_operators, axes=1) + self._drift_terms
        ).flatten(order="F")

    def evaluate_rhs(self, signal_values: Union[List[Array], Array], y: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.
        Args:
            signal_values: either a list [ham_sig_values, dis_sig_values]
                storing the signal values for the Hamiltonian component
                and the dissipator component, or a single array containing
                the total list of signal values.
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns:
            Vectorized RHS of Lindblad equation \dot{\rho} in column-stacking
                convention."""
        if isinstance(signal_values, list):
            if np.any(signal_values[1] != 0):
                signal_values = np.append(signal_values[0], signal_values[1], axis=-1)
            else:
                signal_values = signal_values[0]
        return super().evaluate_rhs(signal_values, y)

    def evaluate_generator(self, signal_values: Union[List[Array], Array]) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.
        Args:
            signal_values: either a list [ham_sig_values, dis_sig_values]
                storing the signal values for the Hamiltonian component
                and the dissipator component, or a single array containing
                the total list of signal values.
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns:
            Vectorized RHS of Lindblad equation \dot{\rho} in column-stacking
                convention."""
        if isinstance(signal_values, list):
            if np.any(signal_values[1] != 0):
                signal_values = np.append(signal_values[0], signal_values[1], axis=-1)
            else:
                signal_values = signal_values[0]
        return super().evaluate_generator(signal_values)


class SparseLindbladCollection(DenseLindbladCollection):
    def __init__(
        self,
        hamiltonian_operators: Array,
        drift: Array,
        dissipator_operators: Optional[Array] = None,
        tol: Optional[int] = 10,
    ):
        r"""Converts an array of Hamiltonian components and signals,
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            drift: If supplied, treated as a constant term to be added to the
                Hamiltonian of the system.
            dissipator_operators: the terms L_j in Lindblad equation.
                (m,n,n) array.
            tol: operator values will be rounded to tol places after the
                decimal place to avoid excess storage of near-zero values
                in sparse format.
        """

        self._hamiltonian_operators = np.empty(shape=hamiltonian_operators.shape[0], dtype="O")
        for i in range(hamiltonian_operators.shape[0]):
            self._hamiltonian_operators[i] = csr_matrix(np.round(hamiltonian_operators[i], tol))
        self.drift = csr_matrix(np.round(drift, tol))
        if dissipator_operators is not None:
            self._dissipator_operators = np.empty(shape=dissipator_operators.shape[0], dtype="O")
            self._dissipator_operators_conj = np.empty_like(self._dissipator_operators)
            for i in range(dissipator_operators.shape[0]):
                self._dissipator_operators[i] = csr_matrix(np.round(dissipator_operators[i], tol))
                self._dissipator_operators_conj[i] = (
                    self._dissipator_operators[i].conjugate().transpose()
                )
            self._dissipator_products = self._dissipator_operators_conj * self._dissipator_operators

    def evaluate_hamiltonian(self, signal_values: Array) -> csr_matrix:
        return np.sum(signal_values * self._hamiltonian_operators, axis=-1) + self.drift

    def evaluate_rhs(self, signal_values: List[Array], y: Array) -> Array:

        hamiltonian_matrix = -1j * self.evaluate_hamiltonian(signal_values[0])  # B matrix

        # For fast matrix multiplicaiton we need to package (n,n) Arrays as (1)
        # Arrays of dtype object, or (k,n,n) Arrays as (k,1) Arrays of dtype object
        y = package_density_matrices(y)

        if self._dissipator_operators is not None:
            dissipators_matrix = (-1 / 2) * np.sum(
                signal_values[1] * self._dissipator_products, axis=-1
            )

            left_mult_contribution = np.squeeze([hamiltonian_matrix + dissipators_matrix] * y)
            right_mult_contribution = np.squeeze(y * [-hamiltonian_matrix + dissipators_matrix])

            # both_mult_contribution[i] = \gamma_i L_i\rho L_i^\dagger performed in array language
            both_mult_contribution = (
                (signal_values[1] * self._dissipator_operators)
                * y
                * self._dissipator_operators_conj
            )
            # sum on i
            both_mult_contribution = np.sum(both_mult_contribution, axis=-1)

            out = left_mult_contribution + right_mult_contribution + both_mult_contribution

        else:
            out = ([hamiltonian_matrix] * y) - (y * [hamiltonian_matrix])
        if len(y.shape) == 2:
            # Very slow step, so avoid if not necessary (or if a better implementation found). Would
            # need to map a (k) Array of dtype object with j^{th} entry a (n,n) Array -> (k,n,n) Array.
            out = unpackage_density_matrices(out.reshape(y.shape[0], 1))

        return out


def package_density_matrices(y: Array) -> Array:
    """Sends a (k,n,n) Array y of density matrices to a
    (k,1) Array of dtype object, where entry [j,0] is
    y[j]. Formally avoids For loops through vectorization.
    Args:
        y: (k,n,n) Array
    Returns:
        Array with dtype object"""
    # As written here, only works for (n,n) Arrays
    obj_arr = np.empty(shape=(1), dtype="O")
    obj_arr[0] = y
    return obj_arr


# Using vectorization with signature, works on (k,n,n) Arrays -> (k,1) Array
package_density_matrices = np.vectorize(package_density_matrices, signature="(n,n)->(1)")


def unpackage_density_matrices(y: Array) -> Array:
    """Inverse function of package_density_matrices,
    Much slower than packaging. Avoid using unless
    absolutely needed (as in case of passing multiple
    density matrices to SparseLindbladCollection.evaluate_rhs)"""
    return y[0]


unpackage_density_matrices = np.vectorize(unpackage_density_matrices, signature="(1)->(n,n)")
