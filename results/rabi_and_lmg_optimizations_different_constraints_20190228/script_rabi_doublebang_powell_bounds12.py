import os
import sys
import numpy as np
import pandas as pd

import logging

if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization


model = 'rabi'
model_parameters = dict(N=100, Omega=100, omega_0=1.)
protocol = 'doublebang'
optimization_method = 'Powell'
parameters_constraints = [-12, 12]

# ------ build and check name for output file
additional_file_name_qualifiers = None
output_file_name = (model + '_' + protocol + '_' +
                    optimization_method.replace('-', '').lower())
if additional_file_name_qualifiers is not None:
    output_file_name += '_' + additional_file_name_qualifiers
filenum = 1
_output_file_name = output_file_name
while os.path.isfile(_output_file_name + '.csv'):
    _output_file_name = output_file_name + '({:02})'.format(filenum)
    filenum += 1
output_file_name = _output_file_name + '.csv'

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
        task='critical point state generation'
    ),
    optimization_specs=dict(
        protocol=protocol,
        optimization_method=optimization_method,
        parameters_constraints=parameters_constraints
    ),
    other_options=dict(
        scan_times=np.linspace(0.1, 4, 100)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

