import os
import sys
import numpy as np
import pandas as pd

import logging

if '../../../' not in sys.path:
    sys.path.append('../../../')
import src.optimization as optimization
import src.protocol_ansatz as protocol_ansatz
from src.utils import autonumber_filename


model = 'lmg'
model_parameters = dict(num_spins=50)
optimization_method = 'Powell'
time_initial = 0.5
time_final = 10
time_subintervals = 200
task=dict(initial_intensity=0, final_intensity=1.5)
parameters_constraints = [-4, 4]

protocol = 'doublebang'

# ------ build and check name for output file
output_file_name = os.path.basename(__file__)[7:-3] + '.csv'
output_file_name = autonumber_filename(output_file_name)

# ------ set up logger
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
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
        task=task
    ),
    optimization_specs=dict(
        protocol=protocol,
        optimization_method=optimization_method,
        parameters_constraints=parameters_constraints
    ),
    other_options=dict(
        scan_times=np.linspace(time_initial, time_final, time_subintervals)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

