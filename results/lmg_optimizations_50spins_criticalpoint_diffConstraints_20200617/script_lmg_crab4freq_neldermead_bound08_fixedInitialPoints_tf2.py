import os
import sys
import numpy as np
import pandas as pd
import logging
if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization
import src.protocol_ansatz as protocol_ansatz
from src.utils import autonumber_filename, basic_logger_configuration

output_file_name = os.path.basename(__file__)[7:-3] + '.csv'
output_file_name = autonumber_filename(output_file_name)
basic_logger_configuration(filename=output_file_name[:-3] + 'log')
logging.info('Output file name will be "{}"'.format(output_file_name))

# ------ start optimization
num_frequencies = 4
protocol = protocol_ansatz.CRABProtocolAnsatz(num_frequencies=num_frequencies)
protocol.generate_rnd_frequencies_each_tf = False
for idx in range(num_frequencies):
    protocol.hyperpars['nuk' + str(idx + 1)] = 0
protocol.fill_hyperpar_value(y0=0, y1=1)

results = optimization.find_best_protocol(
    problem_specification=dict(
        model='lmg',
        model_parameters=dict(num_spins=50),
        task=dict(initial_intensity=0, final_intensity=1)
    ),
    optimization_specs=dict(
        protocol=protocol,
        protocol_options=dict(num_frequencies=num_frequencies),
        optimization_method='Nelder-Mead',
        parameters_constraints=[-8, 8],
        initial_parameters=[0] * (2 * num_frequencies),
        optimization_options=dict(maxiter=1e5, maxfev=1e5,
                                  xatol=1e-8, fatol=1e-8, adaptive=True)
    ),
    other_options=dict(
        scan_times=np.linspace(0.01, 2, 100)
    )
)

# ------ save results to file
results.to_csv(output_file_name)
