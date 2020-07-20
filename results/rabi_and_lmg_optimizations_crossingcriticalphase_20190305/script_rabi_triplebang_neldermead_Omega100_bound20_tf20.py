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

third_time = lambda t: t / 3.
twothirds_time = lambda t: 2 * t / 3.
initial_parameters = [[-1, 1]] * 3 + [third_time, twothirds_time]
# ------ start optimization
results = optimization.find_best_protocol(
    problem_specification=dict(
        model='rabi',
        model_parameters=dict(N=100, Omega=100, omega_0=1.),
        task=dict(initial_intensity=0, final_intensity=10)
    ),
    optimization_specs=dict(
        protocol=protocol_ansatz.TripleBangProtocolAnsatz(),
        optimization_method='Nelder-Mead',
        parameters_constraints=[-20, 20],
        initial_parameters=initial_parameters
    ),
    other_options=dict(
        scan_times=np.linspace(4, 20, 400)
    )
)

# ------ save results to file
results.to_csv(output_file_name)
