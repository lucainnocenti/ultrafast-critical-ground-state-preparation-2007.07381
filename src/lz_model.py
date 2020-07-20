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


class LZModel(TimeDependentHamiltonian):
    def __init__(self, omega_0, td_protocol=None):
        self.name = 'lz'
        self.model_parameters = dict(omega_0=omega_0)
        self.critical_value = 0.
        if td_protocol is not None and isinstance(td_protocol, str):
            self._parse_td_protocol(td_protocol)
        else:
            self.td_protocol = td_protocol

        self.H0 = omega_0 * qutip.sigmax()
        self.H1 = qutip.sigmaz()
