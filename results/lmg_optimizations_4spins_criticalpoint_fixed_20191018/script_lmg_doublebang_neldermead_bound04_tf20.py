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
protocol = 'doublebang'
initial_parameters = [[-1, 1], 'halftime', [-1, 1]]
parameters_constraints = [-4, 4]

# ------ build and check name for output file
output_file_name = src.utils.make_new_data_filename(
    model=model, protocol=protocol,
    optim_method=optimization_method.replace('-', '').lower(),
    bound=parameters_constraints[1],
    other_stuff='tf20'
)
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
results = optimization.find_best_protocol(
    problem_specification=dict(
        model=model,
        model_parameters=model_parameters,
        task='critical point'
    ),
    optimization_specs=dict(
        protocol=protocol,
        optimization_method=optimization_method,
        initial_parameters=initial_parameters,
        parameters_constraints=parameters_constraints
    ),
    other_options=dict(
        scan_times=np.linspace(0.1, 20, 1000)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

