import datetime
import logging
import os
import pickle
import functools
import sys
import numbers
import time

import numpy as np
import pandas as pd
import qutip

from src.utils import ground_state
import src.protocol_ansatz as protocol_ansatz


class TimeDependentHamiltonian:
    def __init__(self, model_parameters):
        """Should involve initialization of core model parameters.
        
        Define time-independent and time-dependent components (`H0` and `H1`),
        name of the model, and time-dependent protocol (as ProtocolAnsatz obj
        instance).
        """
        self.H0 = None
        self.H1 = None
        self.td_protocol = None  # a ProtocolAnsatz object
        self.model_parameters = None
        raise NotImplementedError('Children must override this.')
    
    def hamiltonian(self, parameter):
        """Return the Hamiltonian for a given value of the parameter."""
        return self.H0 + parameter * self.H1
    
    def ground_state(self, parameter):
        return ground_state(self.hamiltonian(parameter))
    
    def _parse_td_protocol(self, td_protocol_name):
        """Input must be a string, parsed to figure out the protocol to use."""
        if td_protocol_name == 'doublebang':
            self.td_protocol = protocol_ansatz.DoubleBangProtocolAnsatz()
        elif td_protocol_name == 'bangramp':
            self.td_protocol = protocol_ansatz.BangRampProtocolAnsatz()
        elif td_protocol_name == 'triplebang':
            self.td_protocol = protocol_ansatz.TripleBangProtocolAnsatz()
        else:
            raise ValueError('Other protocols must be given explicitly.')
    
    def critical_hamiltonian(self):
        return self.hamiltonian(self.critical_value)
    
    def critical_ground_state(self):
        return ground_state(self.critical_hamiltonian)
    
    def evolve_state(self, state, tlist, td_protocol_parameters,
                     return_all_states=False, solver_options=None):
        if self.td_protocol is None:
            raise ValueError('The protocol must be specified first.')
        if isinstance(tlist, numbers.Number):
            tlist = np.linspace(0, tlist, 40)

        protocol = self.td_protocol  # a protocol_ansatz.ProtocolAntatz object
        # we outsource evolving the states to the ProtocolAnsatz object (this
        # is because different protocols might prefer different ways to solve
        # the dynamics)
        return protocol.evolve_state(
            time_independent_ham=self.H0,
            time_dependent_ham=self.H1,
            protocol_parameters=td_protocol_parameters,
            initial_state=state, tlist=tlist,
            return_all_states=return_all_states, solver_options=solver_options
        )

