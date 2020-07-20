import datetime
import os
import pickle
import functools
import sys
import time
import logging
"""
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
"""

import numpy as np
import pandas as pd
import progressbar
import qutip
import qutip.control.pulseoptim as cpo
import scipy

if '../../' not in sys.path:
    sys.path.append('../../')
import src.rabi_model as rabi_model
import src.optimization as optimization
from src.utils import ground_state

# set up output file and dir
output_file_basename = 'rabi_constantfun_scan'
if os.path.isfile(output_file_basename):
    raise ValueError('Change the name of the fucking output file!!')
if os.path.exists(output_file_basename):
    raise ValueError('The directory already exists, there might be something'
                     'wrong!')
os.makedirs(output_file_basename)

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
# define the type of protocol and parameter range
heights_to_try = np.linspace(-20, 20, 200)
times_to_try = np.arange(0.2, 40.2, 0.2)

bar = progressbar.ProgressBar()
for time in bar(times_to_try):
    results = np.zeros(shape=[heights_to_try.shape[0], 3])
    for idx, height in enumerate(heights_to_try):
        hamiltonian = H0 + height * H1
        output_state = optimization.evolve_state(
            hamiltonian=hamiltonian, initial_state=initial_state,
            time=time
        )
        fid = qutip.fidelity(target_state, output_state) ** 2
        results[idx] = [time, height, fid]
    df_to_save = pd.DataFrame(results, columns=['t', 'height', 'fid'])
    outfile_name = os.path.join(output_file_basename, 't{:04.1f}'.format(time))
    df_to_save.to_csv(outfile_name)


