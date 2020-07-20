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
results = optimization.find_best_protocol(
    problem_specification=dict(
        model='rabi',
        model_parameters=dict(N=100, Omega=100, omega_0=1.),
        task=dict(initial_intensity=0, final_intensity=10)
    ),
    optimization_specs=dict(
        protocol='crab',
        protocol_options=dict(num_frequencies=4),
        optimization_method='Nelder-Mead',
        parameters_constraints=[-20, 20]
    ),
    other_options=dict(
        scan_times=np.linspace(4, 20, 400)
    )
)

# ------ save results to file
results.to_csv(output_file_name)
