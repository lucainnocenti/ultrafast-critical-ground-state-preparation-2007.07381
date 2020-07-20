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
num_frequencies = 10
protocol = protocol_ansatz.CRABProtocolAnsatz(num_frequencies=num_frequencies)
protocol.generate_rnd_frequencies_each_tf = False
for idx in range(num_frequencies):
    protocol.hyperpars['nuk' + str(idx + 1)] = 0
protocol.fill_hyperpar_value(y0=-5, y1=0)

results = optimization.find_best_protocol(
    problem_specification=dict(
        model='lz',
        model_parameters=dict(omega_0=1),
        task=dict(initial_intensity=-5, final_intensity=0)
    ),
    optimization_specs=dict(
        protocol=protocol,
        protocol_options=dict(num_frequencies=num_frequencies),
        optimization_method='powell',
        parameters_constraints=[-10, 10],
        initial_parameters=[0] * (2 * num_frequencies)
    ),
    other_options=dict(
        scan_times=np.linspace(0.01, 1, 200)
    )
)

# ------ save results to file
results.to_csv(output_file_name)
