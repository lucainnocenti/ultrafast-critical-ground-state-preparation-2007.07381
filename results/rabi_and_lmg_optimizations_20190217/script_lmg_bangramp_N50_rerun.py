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
import qutip.control.pulseoptim as cpo
import scipy

if '../../' not in sys.path:
    sys.path.append('../../')
import src.lmg_model as lmg_model
import src.optimization as optimization
from src.utils import ground_state


# model parameters
num_spins = 50
optim_objs = lmg_model.prepare_hamiltonian_and_states_for_optimization(
    num_spins=num_spins)
# run optimization
times_to_try = np.linspace(0.01, 1, 100)
results = np.zeros(shape=[len(times_to_try), 6])

for idx, tf in enumerate(times_to_try):
    parametrized_model = functools.partial(
        optimization.make_bangramp_pulse_fun, tf=tf)
    logging.info('Starting optimization for tf={}'.format(tf))
    result = optimization.optimize_model_parameters(
        hamiltonians=[optim_objs['H0'], optim_objs['H1']],
        initial_state=optim_objs['initial_state'],
        target_state=optim_objs['target_state'],
        evolution_time=tf,
        parametrized_model=parametrized_model,
        initial_parameters=[1, tf / 2, 1, 1],
        optimization_method='Powell', stfu=True
    )
    obtained_fidelity = (1 - result.fun)**2
    results[idx] = [tf, obtained_fidelity, *result.x]
    logging.info('    Result: {}'.format(obtained_fidelity))
results = pd.DataFrame(results, columns=['tf', 'fid', 'y0', 't1', 'y1', 'y2'])

results.to_csv('lmg_N50_bangramp_powell_rerun.csv')
