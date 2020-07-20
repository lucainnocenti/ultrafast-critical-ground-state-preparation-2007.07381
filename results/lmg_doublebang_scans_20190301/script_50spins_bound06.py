import sys
import progressbar
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
tf_list = np.linspace(0, 2, 100)
time_ratios_list = np.linspace(0, 1, 100)
bound = [-6, 6]

lmg = lmg_model.LMGModel(num_spins=num_spins)
initial_state = ground_state(lmg.H0)
target_state = ground_state(lmg.hamiltonian(g_value=1.))

results = np.zeros(shape=(len(tf_list) * len(time_ratios_list), 3))

bar = progressbar.ProgressBar()
idx = 0
for tf in bar(tf_list):
    for ratio in time_ratios_list:
        protocol = protocols.DoubleBangProtocolAnsatz()
        protocol.fill_hyperpar_value(tf=tf)
        fun = protocol.time_dependent_fun(np.asarray([bound[1], ratio * tf, bound[0]]))
        output_state = optimization.evolve_state([lmg.H0, [lmg.H1, fun]], initial_state, tf)
        results[idx] = [tf, ratio, qutip.fidelity(output_state, target_state)**2]
        idx += 1
results = pd.DataFrame(results, columns=['tf', 'ratio', 'fid'])

results.to_csv('scan_{}spins_bound{}.csv'.format(num_spins, bound[1]))
