import datetime
import os
import pickle
import sys
import time
import functools
import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

import numpy as np
import pandas as pd
import progressbar
import qutip
# import qutip.control.pulseoptim as cpo
# import qutip.logging_utils as logging
import scipy

if '../../' not in sys.path:
    sys.path.append('../../')
import src.rabi_model as rabi_model
import src.optimization as optimization
from src.utils import ground_state


# model parameters
N = 100
Omega = 100
omega_0 = 1
lambda_c = np.sqrt(Omega * omega_0) / 2.
# build Hamiltonians
H0 = rabi_model.QRM_free_term(N, Omega, omega_0)
H1 = rabi_model.QRM_interaction_term(N)
# compute initial and target states
initial_state = ground_state(H0)
target_state = ground_state(H0 + lambda_c * H1)
# run optimization
times_to_try = np.linspace(0.1, 4, 100)
num_frequencies = 10
num_CRAB_pars = 3 * num_frequencies
results = np.zeros(shape=[len(times_to_try), 2 + num_CRAB_pars])

for idx, tf in enumerate(times_to_try):
    # crab hyperparameters
    nuk = np.random.rand(num_frequencies) - 0.5
    nuk = nuk * tf / num_frequencies

    parametrized_model = functools.partial(
        optimization.make_CRAB_final_ramp_fun,
        nuk=nuk, tf=tf, y0=0, y1=lambda_c
    )
    logging.info('Starting optimization for tf={}'.format(tf))
    result = optimization.optimize_model_parameters(
        hamiltonians=[H0, H1],
        initial_state=initial_state, target_state=target_state,
        evolution_time=tf,
        parametrized_model=parametrized_model,
        initial_parameters=np.random.randn(2 * num_frequencies),
        optimization_method='Nelder-Mead', stfu=True
    )
    results[idx] = [tf, (1 - result.fun)**2, *nuk, *result.x]
    logging.info('    Result: {}'.format(1 - result.fun))

columns_strings = ['tf', 'fid'] + ['nu' + str(k) for k in range(num_frequencies)]
columns_strings += ['A' + str(k) for k in range(num_frequencies)]
columns_strings += ['B' + str(k) for k in range(num_frequencies)]

results = pd.DataFrame(results, columns=columns_strings)
results.to_csv('rabi_Omega100_crab_10freq_neldermead.csv')

