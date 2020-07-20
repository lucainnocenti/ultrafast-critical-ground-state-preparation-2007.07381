import datetime
import logging
import importlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import qutip


def ground_state(H):
    """Shortcut to compute the ground state of an Hamiltonian.
    
    Parameters
    ----------
    H : qutip.Qobj matrix
    """
    return H.eigenstates(eigvals=1)[1][0]

def plot(fun, times, *args, **kwargs):
    plt.figure()
    plt.plot(times, [fun(t) for t in times], *args, **kwargs)


def autonumber_filename(filename):
    base, ext = os.path.splitext(filename)
    if not os.path.isfile(filename):
        return filename
    idx = 0
    while os.path.isfile('{}({:02}){}'.format(base, idx, ext)):
        idx += 1
    filename = base + '({:02})'.format(idx) + ext
    return filename


def timestamp():
    return datetime.datetime.now().strftime('%y%m%dh%Hm%Ms%S')


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple, np.ndarray)):
            for j in flatten(i):
                yield j
        else:
            yield i


def make_new_data_filename(model, protocol, optim_method,
                           bound=None, other_stuff=None, num_frequencies=None):
    """Build the string for a new dataset filename.
    
    NOTE: The code to rename files after the (1) has been created doesn't
    actualy work (it currently just appends new numbers). To edit with some
    re.match sweetness.
    """
    if protocol == 'crab' and num_frequencies is None:
        print('BEWARE, you specified CRAB but did not put the number of'
              'frequencies in the filename. If this is on purpose, carry on.')
    if protocol == 'crab' and num_frequencies is not None:
        output_str = '{}_crab{}freq_{}'.format(model, num_frequencies, optim_method)
    else:
        output_str = '{}_{}_{}'.format(model, protocol, optim_method)
    if bound is not None:
        output_str += '_bound{:02}'.format(bound)
    if other_stuff is not None:
        output_str += '_' + other_stuff
    # add the extension
    output_str += '.csv'
    # check whether filename already existed
    filenum = 1
    while os.path.isfile(output_str):
        output_str = output_str[:-4] + '({:02}).csv'.format(filenum)
        filenum += 1

    return output_str


def basic_logger_configuration(filename=None, toconsole=False,
                               reset=False, level=logging.DEBUG):
    if filename is None and not toconsole:
        raise ValueError('At least one of tofile and toconsole must be true.')
    if reset:
        logging.shutdown()
        importlib.reload(logging)
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                     "[%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)
    if filename is not None and isinstance(filename, str):
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(level)
        rootLogger.addHandler(fileHandler)
    if toconsole:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

def fidelity(first_vec, second_vec):
    """Compute the (UNSQUARED) fidelity between two ket states.

    Simply the absolute value of the overlap of the vectors.

    We need this because qutip.fidelity IS NOT RELIABLE to high precisions, due
    to its always converting kets into dms and using the general formula to
    compute the fidelity.
    See also the issue on github: https://github.com/qutip/qutip/issues/925.
    """
    if isinstance(first_vec, qutip.Qobj):
        first_vec = first_vec.full()
    if isinstance(second_vec, qutip.Qobj):
        second_vec = second_vec.full()
    
    return np.abs(np.vdot(first_vec, second_vec))

