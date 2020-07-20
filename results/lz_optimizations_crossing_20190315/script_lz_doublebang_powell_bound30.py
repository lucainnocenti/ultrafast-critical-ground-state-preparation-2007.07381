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
        model='lz',
        model_parameters=dict(omega_0=1),
        task=dict(initial_intensity=-5, final_intensity=5)
    ),
    optimization_specs=dict(
        protocol='doublebang',
        optimization_method='Powell',
        parameters_constraints=[-30, 30],
        initial_parameters=[[-2, 2], 'halftime', [-2, 2]]
    ),
    other_options=dict(
        scan_times=np.linspace(0.01, 10, 400)
    )
)

# ------ save results to file
results.to_csv(output_file_name)
