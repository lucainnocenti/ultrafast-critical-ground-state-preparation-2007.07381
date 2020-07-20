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
model_parameters = dict(num_spins=20)
optimization_method = 'Powell'
protocol = protocol_ansatz.TripleBangProtocolAnsatz()

parameters_constraints = [-4, 4]
third_time = lambda t: t / 3.
twothirds_time = lambda t: 2 * t / 3.
initial_parameters = [[-1, 1]] * 3 + [third_time, twothirds_time]

# ------ build and check name for output file
output_file_name = src.utils.make_new_data_filename(
    model=model, protocol=protocol,
    optim_method=optimization_method.replace('-', '').lower(),
    bound=parameters_constraints[1],
    other_stuff='tf20_precise'
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
        parameters_constraints=parameters_constraints,
        optimization_options=dict(xtol=1e-16, ftol=1e-16, disp=False)
    ),
    other_options=dict(
        scan_times=np.linspace(0.1, 2, 100)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

