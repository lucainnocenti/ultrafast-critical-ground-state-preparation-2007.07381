import sys
import os
import progressbar
import logging
import numpy as np
import pandas as pd
import qutip

if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization
import src.rabi_model as rabi_model
import src.protocol_ansatz as protocols
from src.utils import ground_state

N = 100
Omega = N
tf_list = np.linspace(0, 2, 200)
time_ratios_list = np.linspace(0, 1, 200)
heights_list = np.asarray([40, 60, 80, 100])

rabi = rabi_model.RabiModel(N=N, Omega=Omega)
initial_state = ground_state(rabi.H0)
target_state = ground_state(rabi.hamiltonian(lambda_=np.sqrt(Omega) / 2.))


# ------ set up logger
output_file_name = 'scan_' + os.path.basename(__file__)[7:-3]
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
fileHandler = logging.FileHandler(output_file_name + '.log')
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.INFO)
rootLogger.addHandler(fileHandler)
logging.info('Output file name will be "{}"'.format(output_file_name))

# ------ GOGOGOGO
logging.info('Starting super iteration loop')
for height_idx, height in enumerate(heights_list):
    logging.info('Working on height {}/{}'.format(height_idx,
                                                  len(heights_list)))
    results = np.zeros(shape=(len(tf_list) * len(time_ratios_list), 3))
    idx = 0
    for out_idx, tf in enumerate(tf_list):
        logging.info('Starting iteration {}/{}'.format(out_idx + 1, len(list(tf_list))))
        for ratio in time_ratios_list:
            protocol = protocols.DoubleBangProtocolAnsatz()
            protocol.fill_hyperpar_value(tf=tf)
            fun = protocol.time_dependent_fun(np.asarray([height, ratio * tf, 0]))
            output_state = optimization.evolve_state([rabi.H0, [rabi.H1, fun]],
                                                     initial_state, tf)
            results[idx] = [tf, ratio, qutip.fidelity(output_state, target_state)**2]
            idx += 1
    results = pd.DataFrame(results, columns=['tf', 'ratio', 'fid'])

    results.to_csv(output_file_name + '_height{:03}.csv'.format(height))
