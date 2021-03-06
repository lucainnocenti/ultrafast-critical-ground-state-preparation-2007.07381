import os
import sys
import numpy as np
import pandas as pd

import logging

if '../../' not in sys.path:
    sys.path.append('../../')
import src.utils
import src.optimization as optimization
import src.protocol_ansatz as protocol_ansatz


model = 'lmg'
model_parameters = dict(num_spins=4)
optimization_method = 'Nelder-Mead'
protocol = 'crab'
num_frequencies = 4

parameters_constraints = [-4, 4]
initial_parameters = [[-0.1, 0.1]] * (2 * num_frequencies)

# ------ build and check name for output file
output_file_name = src.utils.make_new_data_filename(
    model=model, protocol=protocol, num_frequencies=num_frequencies,
    optim_method=optimization_method.replace('-', '').lower(),
    bound=parameters_constraints[1],
    other_stuff='tf20_precise'
)
# ------ set up logger
src.utils.basic_logger_configuration(filename=output_file_name[:-4] + '.log')
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
#                                  "[%(levelname)-5.5s]  %(message)s")
# rootLogger = logging.getLogger()
# rootLogger.setLevel(logging.DEBUG)
# # consoleHandler = logging.StreamHandler()
# # consoleHandler.setFormatter(logFormatter)
# # rootLogger.addHandler(consoleHandler)
# fileHandler = logging.FileHandler(output_file_name[:-4] + '.log')
# fileHandler.setFormatter(logFormatter)
# fileHandler.setLevel(logging.DEBUG)
# rootLogger.addHandler(fileHandler)


logging.info('Output file name will be "{}"'.format(output_file_name))

# ------ start optimization
results = optimization.find_best_protocol(
    problem_specification=dict(
        model=model,
        model_parameters=model_parameters,
        task='critical point'
    ),
    optimization_specs=dict(
        protocol=protocol, protocol_options=dict(num_frequencies=num_frequencies),
        optimization_method=optimization_method,
        initial_parameters=initial_parameters,
        parameters_constraints=parameters_constraints,
        optimization_options=dict(xatol=1e-16, fatol=1e-16, disp=True)
    ),
    other_options=dict(
        scan_times=np.linspace(2, 3.2, 500)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

