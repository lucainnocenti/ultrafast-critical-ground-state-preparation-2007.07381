import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import qutip
#QuTiP control modules
import scipy
import scipy.linalg


from src.utils import ground_state
from src.hamiltonians import TimeDependentHamiltonian


def _symmetry_sector_indices(N, which='down'):
    if which == 'down':
        list1 = np.arange(1, 2 * N, 4)
        list2 = np.arange(2, 2 * N, 4)
        # indices = [item for pair in zip(list1, list2) for item in pair]
        indices = [None] * (len(list1) + len(list2))
        indices[::2] = list1
        indices[1::2] = list2
        return indices
    elif which == 'up':
        list1 = np.arange(3, 2 * N, 4)
        list2 = np.arange(4, 2 * N, 4)
        indices = [None] * (len(list1) + len(list2))
        indices[::2] = list1
        indices[1::2] = list2
        indices = [0] + indices
        return indices


def QRM_free_term(N, Omega, omega_0, symmetry_sector='down'):
    """Free term of the QRM Hamiltonian.
    
    Parameters
    ----------
    N : int
        The dimension of the truncated Fock space
    symmetry_sector : int
        If 'full, the full Hamiltonian is returned.
        If 'down', the Hamiltonian is projected in the spin down sector.
        If 'up', the Hamiltonian is projected in the spin up sector.
    Returns
    -------
    A qutip.qObj matrix-like object.
    """
    H0 = Omega / 2 * qutip.tensor(qutip.qeye(N), qutip.sigmaz())
    H0 += omega_0 * qutip.tensor(qutip.create(N) * qutip.destroy(N), qutip.qeye(2))
    if symmetry_sector == 'full':
        return H0
    elif symmetry_sector == 'up' or symmetry_sector == 'down':
        indices = _symmetry_sector_indices(N=N, which=symmetry_sector)
        return qutip.Qobj(H0.full()[np.ix_(indices, indices)])
    else:
        raise ValueError('symmetry_sector must be `up`, `down`, or `full`.')


def QRM_interaction_term(N, symmetry_sector='down'):
    Hint = -qutip.tensor(qutip.create(N) + qutip.destroy(N), qutip.sigmax())
    if symmetry_sector == 'full':
        return Hint
    elif symmetry_sector == 'up' or symmetry_sector == 'down':
        indices = _symmetry_sector_indices(N=N, which=symmetry_sector)
        return qutip.Qobj(Hint.full()[np.ix_(indices, indices)])
    else:
        raise ValueError('symmetry_sector must be `up`, `down`, or `full`.')


def QRM_full(N, Omega, omega_0, lambda_, symmetry_sector='down'):
    """Full QRM Hamiltonian."""
    H = QRM_free_term(N, Omega, omega_0, symmetry_sector=symmetry_sector)
    H += lambda_ * QRM_interaction_term(N, symmetry_sector=symmetry_sector)
    return H


def parity_op(N):
    """Compute the parity operator for the QRM model.

    This assumes that the state lives in the FULL space (so that it was not
    already projected in a parity subspace).
    """
    operator = (qutip.tensor(qutip.create(N) * qutip.destroy(N), qutip.qeye(2)) + 
                qutip.tensor(qutip.qeye(N), (qutip.sigmaz() + qutip.qeye(2)) / 2))
    operator = scipy.linalg.expm(1j * np.pi * operator.full())
    operator = qutip.Qobj(operator)
    operator.dims = [[N, 2], [N, 2]]
    return operator


class RabiModel(TimeDependentHamiltonian):
    def __init__(self, N, Omega, omega_0=1.,
                 symmetry_sector='down', td_protocol=None):
        self.name = 'rabi'
        self.model_parameters = dict(N=N, Omega=Omega, omega_0=omega_0)
        self.critical_value = np.sqrt(Omega * omega_0) / 2.
        if td_protocol is not None and isinstance(td_protocol, str):
            self._parse_td_protocol(td_protocol)
        else:
            self.td_protocol = td_protocol

        self.H0 = QRM_free_term(N=N, Omega=Omega, omega_0=omega_0,
                                symmetry_sector=symmetry_sector)
        self.H1 = QRM_interaction_term(N=N, symmetry_sector=symmetry_sector)
