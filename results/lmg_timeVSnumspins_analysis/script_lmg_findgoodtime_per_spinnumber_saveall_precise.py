import os
import sys
import numpy as np
import pandas as pd

import logging

if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization


model = 'lmg'
protocol = 'doublebang'
optimization_method = 'Powell'

# ------ build and check name for output file
output_file_name = 'goodTime_vs_numSpins_precise.csv'

# ------ set up logger
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)
fileHandler = logging.FileHandler(output_file_name[:-4] + '.log')
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
rootLogger.addHandler(fileHandler)


logging.info('Output file name will be "{}"'.format(output_file_name))

# ------ start optimization
num_spins_list = np.arange(50, 260, 10)
tf_results = np.zeros(shape=(len(num_spins_list), 6))  # will contain (num_spin, tf)
for idx, num_spins in enumerate(num_spins_list):
    logging.info('Starting optimization with {} spins'.format(num_spins))
    results = optimization.find_best_protocol(
        problem_specification=dict(
            model=model,
            model_parameters=dict(num_spins=num_spins),
            task='critical point'
        ),
        optimization_specs=dict(
            protocol=protocol,
            optimization_method=optimization_method,
            optimization_options=dict(xtol=1e-8, ftol=1e-8, disp=True)
        ),
        other_options=dict(
            scan_times=np.arange(1, 2, 0.01),
            stopping_condition=0.999
        )
    )
    tf_results[idx][0] = num_spins
    tf_results[idx][1:] = results.iloc[-1]

tf_results = pd.DataFrame(tf_results, columns=['num_spins', 'tf', 'fid', 'y0',
                                               't1', 'y1'])
# ------ save results to file
tf_results.to_csv(output_file_name)

