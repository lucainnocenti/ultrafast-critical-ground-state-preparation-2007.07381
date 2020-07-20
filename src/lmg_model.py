import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import scipy
import qutip

from src.utils import ground_state
from src.hamiltonians import TimeDependentHamiltonian


def J_plus_operator(num_spins, total_j):
    dim_space = int(2 * total_j) + 1
    Jp_matrix = np.zeros(shape=[dim_space] * 2)
    for m in range(int(2 * total_j)):
        m_orig = m - total_j
        Jp_matrix[m + 1, m] = np.sqrt(total_j * (total_j + 1) -
                                      m_orig * (m_orig + 1))
    return qutip.Qobj(Jp_matrix)


def J_minus_operator(num_spins, total_j):
    dim_space = int(2 * total_j) + 1
    Jm_matrix = np.zeros(shape=[dim_space] * 2)
    for m in range(1, int(2 * total_j) + 1):
        m_orig = m - total_j
        Jm_matrix[m - 1, m] = np.sqrt(total_j * (total_j + 1) -
                                      m_orig * (m_orig - 1))
    return qutip.Qobj(Jm_matrix)


def Jx_operator(num_spins, total_j):
    return (J_plus_operator(num_spins, total_j) +
            J_minus_operator(num_spins, total_j))


def Jz_operator(num_spins, total_j):
    return qutip.Qobj(np.diag(np.arange(-total_j, total_j + 1)))


def LMG_free_term(num_spins, total_j=None, symmetry_sector='first'):
    if total_j is None:
        total_j = num_spins / 2
    op = Jz_operator(num_spins, total_j)
    if symmetry_sector == 'full':
        return op
    elif symmetry_sector == 'first':
        return qutip.Qobj(op.full()[::2, ::2])
    elif symmetry_sector == 'second':
        return qutip.Qobj(op.full()[1::2, 1::2])
    else:
        raise ValueError('Only accepted values are `full`, `first`, `second`.')


def LMG_interaction_term(num_spins, g_value=1., total_j=None,
                         symmetry_sector='first'):
    """Interaction term of the LMG Hamiltonian.

    Given in the standard form - g / 4N * Sx^2
    """
    if total_j is None:
        total_j = num_spins / 2
    Jx_squared = Jx_operator(num_spins, total_j) ** 2
    op = - g_value / (4 * num_spins) * Jx_squared
    if symmetry_sector == 'full':
        return op
    elif symmetry_sector == 'first':
        return qutip.Qobj(op.full()[::2, ::2])
    elif symmetry_sector == 'second':
        return qutip.Qobj(op.full()[1::2, 1::2])
    else:
        raise ValueError('Only accepted values are `full`, `first`, `second`.')


def LMG_full_hamiltonian(num_spins, g_value=1., total_j=None,
                         symmetry_sector='first'):
    op = LMG_free_term(num_spins, total_j, symmetry_sector=symmetry_sector)
    op += LMG_interaction_term(num_spins, g_value, total_j,
                               symmetry_sector=symmetry_sector)
    return op
            

def prepare_hamiltonian_and_states_for_optimization(num_spins, total_j=None,
                                                    symmetry_sector='first'):
    H0 = LMG_free_term(num_spins, total_j=total_j,
                       symmetry_sector=symmetry_sector)
    H1 = LMG_interaction_term(num_spins, g_value=1., total_j=total_j,
                              symmetry_sector=symmetry_sector)
    initial_state = ground_state(H0)
    target_state = ground_state(H0 + H1)
    return dict(H0=H0, H1=H1,
                initial_state=initial_state, target_state=target_state)


class LMGModel(TimeDependentHamiltonian):
    def __init__(self, num_spins, total_j=None,
                 symmetry_sector='first', td_protocol=None):
        self.name = 'lmg'
        if total_j is None:
            total_j = num_spins / 2
        self.model_parameters = dict(num_spins=num_spins, total_j=total_j)
        self.critical_value = 1.
        if td_protocol is not None and isinstance(td_protocol, str):
            self._parse_td_protocol(td_protocol)
        else:
            self.td_protocol = td_protocol

        self.H0 = LMG_free_term(num_spins, symmetry_sector=symmetry_sector)
        self.H1 = LMG_interaction_term(num_spins, total_j=total_j,
                                       symmetry_sector=symmetry_sector)
