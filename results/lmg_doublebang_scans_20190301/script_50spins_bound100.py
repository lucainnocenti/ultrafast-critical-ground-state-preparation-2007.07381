import sys
import progressbar
import logging
import numpy as np
import pandas as pd
import qutip

if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization
import src.lmg_model as lmg_model
import src.protocol_ansatz as protocols
from src.utils import ground_state

num_spins = 50
tf_list = np.linspace(0, 1, 200)
time_ratios_list = np.linspace(0, 1, 400)
bound = [-100, 100]

lmg = lmg_model.LMGModel(num_spins=num_spins)
initial_state = ground_state(lmg.H0)
target_state = ground_state(lmg.hamiltonian(g_value=1.))


# ------ set up logger
output_file_name = 'scan_{}spins_bound{}.csv'.format(num_spins, bound[1])
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler(output_file_name[:-4] + '.log')
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
rootLogger.addHandler(fileHandler)
logging.info('Output file name will be "{}"'.format(output_file_name))

# ------ GOGOGOGO
logging.info('Starting iteration loop')
results = np.zeros(shape=(len(tf_list) * len(time_ratios_list), 3))
idx = 0
for out_idx, tf in enumerate(tf_list):
    logging.info('Starting iteration {}/{}'.format(out_idx + 1, len(list(tf_list))))
    for ratio in time_ratios_list:
        protocol = protocols.DoubleBangProtocolAnsatz()
        protocol.fill_hyperpar_value(tf=tf)
        fun = protocol.time_dependent_fun(np.asarray([bound[1], ratio * tf, bound[0]]))
        output_state = optimization.evolve_state([lmg.H0, [lmg.H1, fun]], initial_state, tf)
        results[idx] = [tf, ratio, qutip.fidelity(output_state, target_state)**2]
        idx += 1
results = pd.DataFrame(results, columns=['tf', 'ratio', 'fid'])

results.to_csv('scan_{}spins_bound{}.csv'.format(num_spins, bound[1]))
