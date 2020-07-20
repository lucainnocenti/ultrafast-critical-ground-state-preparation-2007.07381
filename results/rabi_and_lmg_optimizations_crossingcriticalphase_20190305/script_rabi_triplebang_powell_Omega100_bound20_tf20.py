import os
import sys
import numpy as np
import pandas as pd

import logging

if '../../' not in sys.path:
    sys.path.append('../../')
import src.optimization as optimization
import src.protocol_ansatz as protocol_ansatz


model = 'rabi'
model_parameters = dict(N=100, Omega=100, omega_0=1)
optimization_method = 'Powell'
time_initial = 0.5
time_final = 20
time_subintervals = 200
task=dict(initial_intensity=0, final_intensity=10)
parameters_constraints = [-20, 20]

protocol = protocol_ansatz.TripleBangProtocolAnsatz()
third_time = lambda t: t / 3.
twothirds_time = lambda t: 2 * t / 3.
initial_parameters = [[-1, 1]] * 3 + [third_time, twothirds_time]

# ------ build and check name for output file
output_file_name = os.path.basename(__file__)[7:-3]
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
        parameters_constraints=parameters_constraints,
        initial_parameters=initial_parameters
    ),
    other_options=dict(
        scan_times=np.linspace(time_initial, time_final, time_subintervals)
    )
)

# ------ save results to file
results.to_csv(output_file_name)

