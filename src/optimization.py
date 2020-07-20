import datetime
import logging
import os
import pickle
import functools
import sys
import pickle
import numbers
import time

import numpy as np
import pandas as pd
import progressbar
import qutip
#QuTiP control modules
import qutip.control.pulseoptim as cpo
# import qutip.logging_utils as logging
import scipy

import src.rabi_model as rabi_model
import src.lmg_model as lmg_model
import src.lz_model as lz_model
from src.hamiltonians import TimeDependentHamiltonian
from src.utils import ground_state, autonumber_filename, timestamp, fidelity
from src.protocol_ansatz import (
    DoubleBangProtocolAnsatz, BangRampProtocolAnsatz, CRABProtocolAnsatz)


def evolve_state(hamiltonian, initial_state, tlist,
                            return_all_states=False):
    """Evolve state with (generally time-dependent) hamiltonian.

    This is a wrapper around `qutip.mesolve`, to which `hamiltonian` and
    `initial_state` are directly fed.

    Parameters
    ----------
    hamiltonian : list of qutip objects
        The Hamiltonian specification in `qutip.mesolve` is the
        "function based" one, as per qutip.mesolve documentation. This means
        that `hamiltonian` is to be given as a list of constant Hamiltonians,
        each one pairs with a time-dependent (numeric) coefficient.
        The simplest example would be `hamiltonian = [H0, [H1, coeffFun]]`.
    initial_state : qutip.Qobj
    time : float or list of floats
        If a single number, it is divided into a number of subintervals and
        the result used as input for `qutip.mesolve`. If a list of numbers,
        it is directly fed to `qutip.mesolve`.
    """
    if isinstance(tlist, numbers.Number):
        tlist = np.linspace(0, tlist, 40)

    try:
        evolving_states = qutip.mesolve(hamiltonian, initial_state, tlist)
    except Exception:
        error_data_filename = 'evolution_error_details_{}.pickle'.format(
            timestamp())
        error_data_filename = os.path.join(os.getcwd(), error_data_filename)
        error_data_filename = autonumber_filename(error_data_filename)
        logging.info('Something went wrong while trying to evolve from\n'
                     'initial_state={}\nwith hamiltonian={}.\nSaving data to '
                     'reproduce in "{}".'.format(initial_state, hamiltonian,
                                                 error_data_filename))
        with open(error_data_filename, 'wb') as fh:
            if len(hamiltonian) == 2 and len(hamiltonian[1]) == 2:
                pulse_samples = [hamiltonian[1][1](t)
                                 for t in np.linspace(0, time, 100)]
                hamiltonian = (hamiltonian[0], hamiltonian[1][0])
            data = dict(hamiltonian=hamiltonian,
                        initial_state=initial_state, times_list=tlist)
            data = data.update(dict(pulse_samples=pulse_samples))
            print('data: {}'.format(data))
            pickle.dump(data, fh)
        raise

    if return_all_states:
        return evolving_states.states
    else:
        return evolving_states.states[-1]


def evolve_adiabatically(initial_hamiltonian, final_hamiltonian, tlist,
                         return_all_states=False):
    """Evolve the gs of an Hamiltonian towards that of another."""
    if isinstance(tlist, numbers.Number):
        tlist = np.linspace(0, tlist, 40)

    delta_ham = final_hamiltonian - initial_hamiltonian
    def linear_ramp(t, *args):
        return t / tlist[-1]
    H = [initial_hamiltonian, [delta_ham, linear_ramp]]

    initial_state = ground_state(initial_hamiltonian)
    return evolve_state(hamiltonian=H, initial_state=initial_state,
                        tlist=tlist,
                        return_all_states=return_all_states)


# def _optimize_model_parameters(
#         hamiltonians, initial_state, target_state,
#         evolution_time, protocol,
#         initial_parameters,
#         optimization_method='Nelder-Mead',
#         optimization_options=None
#     ):
#     """LEGACY VERSION OF THIS FUNCTION, KEPT ONLY TEMPORARILY.
    
#     Optimize a model ansatz with respect to its parameters.

#     Input and output states are fixed, and so is the parametrized form of the
#     protocol. Optimize the protocol-defining parameters to maximize the
#     fidelity between initial_state and target_state.

#     The total evolution time is generally fixed through the 

#     Parameters
#     ----------
#     hamiltonians : pair of qutip.Qobj hamiltonians
#         Pair [H0, H1] where H0 is the time-independent component of the overall
#         Hamiltonian (the free term), and H1 the time-independent component of
#         the time-dependent part of the overall Hamiltonian (the interaction
#         term).
#         During the optimization, different time-dependent functions will be
#         attached to H1, according to the user-specified model.
#         NOTE: Clearly, this restricts the use of this function to only
#               a specific class of QOC problems.
#     initial_state : qutip.Qobj state
#         For every protocol tried, this state is evolved through the
#         corresponding time-dependent Hamiltonian and compared with the target.
#     target_state : qutip.Qobj state
#         As above: the output state is for every protocol compared with this
#         one (via fidelity).
#     evolution_time : float
#         The total evolution time. This could technically be specified through
#         the parametrized model, but it's just easier for now to have it as a
#         separate argument.
#     protocol : ProtocolAnsatz instance
#         Function taking as input a numpy array, and giving as output a
#         real-to-real function returning, for a particular protocol, the
#         interaction value corresponding to a given time.
#     initial_parameters : list of floats or 1D np.array
#         The optimization will proceed from this initial value of the protocol
#         parameters.
#     optimization_method : string
#         Passed over to scipy.optimize.minimize
#     optimization_options : dict
#         Passed over to scipy.optimize.minimize as the `options` parameteres.
#         It is to be used to specify method-specific options.
#     """
#     BIG_BAD_VALUE = 200  # yes, I pulled this number out of my ass

#     def fidelity_vs_model_parameters(pars):
#         # assign very high cost if outside of the boundaries (we have to do
#         # this because for some reason scipy does not support boundaries with
#         # Nelder-Mead and Powell)

#         if not protocol.are_pars_in_boundaries(pars):
#             # this checks that the parameters are within boundaries (not the
#             # overall protocol, which is checked later)
#             return BIG_BAD_VALUE

#         # build model, hamiltonian, and compute output state
#         time_dependent_fun = protocol.time_dependent_fun(pars)

#         H = [hamiltonians[0], [hamiltonians[1], time_dependent_fun]]
#         output_state = evolve_state(H, initial_state, evolution_time)
#         # check if we stepped out of the overall boundaries (if any) during
#         # the evolution
#         if protocol.total_height_constraint is not None:
#             if protocol.out_of_boundaries:
#                 protocol.out_of_boundaries = False
#                 return BIG_BAD_VALUE
#         # compute and return infidelity (because scipy gives you MINimize)
#         return 1 - fidelity(output_state, target_state)

#     # run the actual optimisation
#     logging.info('Starting optimization for tf={}'.format(evolution_time))
#     logging.debug('Optimization method: {}'.format(optimization_method))
#     logging.debug('Optimization options: {}'.format(optimization_options))
#     logging.info('Initial parameter values: {}'.format(initial_parameters))
#     result = scipy.optimize.minimize(
#         fun=fidelity_vs_model_parameters,
#         x0=initial_parameters,
#         method=optimization_method,
#         options=optimization_options
#         # bounds=parameters_constraints  # not accepted for Powell and NM
#     )
#     logging.info('Final fidelity: {}'.format((1 - result.fun)**2))
#     logging.info('Final parameters: {}'.format(result.x))
#     result.data = dict(initial_pars=initial_parameters)
#     return result


def optimize_model_parameters(
        hamiltonian, initial_state, target_state,
        tlist, initial_parameters,
        optimization_method='Nelder-Mead',
        optimization_options=None, solver_options=None
    ):
    """Optimize a model ansatz with respect to its parameters.

    Input and output states are fixed, and so is the parametrized form of the
    protocol. Optimize the protocol-defining parameters to maximize the
    fidelity between initial_state and target_state.
    NOTE: How the dynamics is actually solved is deferred to the hamiltonian
          object (which usually defers it to the protocol_ansatz object)

    Parameters
    ----------
    hamiltonian : TimeDependentHamiltonian instance
    initial_state : qutip.Qobj state
        For every protocol tried, this state is evolved through the
        corresponding time-dependent Hamiltonian and compared with the target.
    target_state : qutip.Qobj state
        As above: the output state is for every protocol compared with this
        one (via fidelity).
    tlist : float
        The total evolution time. This could technically be specified through
        the parametrized model, but it's just easier for now to have it as a
        separate argument.
    initial_parameters : list_like of floats
        The optimization will proceed from this initial value of the protocol
        parameters.
        NOTE: We don't support ranges or strings here to have flexible or
              random initial parameter values. For that you need to work with
              the higher level interface `find_best_protocol`.
    optimization_method : string
        Passed over to scipy.optimize.minimize
    optimization_options : dict
        Passed over to scipy.optimize.minimize as the `options` parameteres.
        It is to be used to specify method-specific options.
    solver_options : dict
        These are options passed to the solver (usually qutip.mesolve)
    
    Returns
    -------
    The result object returned by scipy.optimize.minimize (essentially a dict
    containing the optimisation data).

    The final value of the cost function is stored in `result.fun`. THIS
    EQUALS (1 - F) WITH F THE *NON*-SQUARED FIDELITY. You need to compute
    1 - (1 - fun)**2 to obtain the infidelity.

    `result.x` contains the optimised values of the protocol parameters.
    """
    BIG_BAD_VALUE = 200  # yes, I pulled this number out of my ass
    if isinstance(tlist, numbers.Number):
        tlist = np.linspace(0, tlist, 40)

    # hamiltonian.td_protocol is a protocol_ansatz object. All of the hyperpars
    # should already have been fixed, only exception being the total time
    if hamiltonian.td_protocol is None:
        raise ValueError('Did you forget to specify a protocol pal?')
    protocol = hamiltonian.td_protocol
    protocol.fill_hyperpar_value(tf=tlist[-1])
    def fidelity_vs_model_parameters(pars):
        # assign very high cost if outside of the boundaries (we have to do
        # this because for some reason scipy does not support boundaries with
        # Nelder-Mead and Powell)

        # check that the parameters are within boundaries (not the overall
        # height of the protocol, which is checked later)
        if not protocol.are_pars_in_boundaries(pars):
            return BIG_BAD_VALUE

        # evolve state
        output_state = hamiltonian.evolve_state(
            state=initial_state, tlist=tlist,
            td_protocol_parameters=pars, return_all_states=False,
            solver_options=solver_options
        )

        # check if we stepped out of the overall boundaries (if any) during
        # the evolution
        if protocol.total_height_constraint is not None:
            if protocol.out_of_boundaries:
                protocol.out_of_boundaries = False
                return BIG_BAD_VALUE
        # compute and return infidelity (because scipy gives you MINimize)
        return 1 - fidelity(output_state, target_state)

    # run the actual optimisation
    logging.info('Starting optimization for tf={}'.format(tlist[-1]))
    logging.debug('Optimization method: {}'.format(optimization_method))
    logging.debug('Optimization options: {}'.format(optimization_options))
    logging.info('Initial parameter values: {}'.format(initial_parameters))
    result = scipy.optimize.minimize(
        fun=fidelity_vs_model_parameters,
        x0=initial_parameters,
        method=optimization_method,
        options=optimization_options
        # bounds=parameters_constraints  # not accepted for Powell and NM
    )
    logging.info('Final fidelity: {}'.format((1 - result.fun)**2))
    logging.info('Final parameters: {}'.format(result.x))
    result.data = dict(initial_pars=initial_parameters)
    return result


def _parse_initial_parameters(initial_parameters, tf):
    """Used by _optimize_model_parameters_scan_times.
    
    Does the following to the input list:
        - numbers are kept without changes
        - strings are kept without changes (will be parsed later)
        - anything else is assumed to be a two-element tuple, and in this case
          a random number in the given range is generated and put instead of
          the tuple.
    """
    parsed_initial_pars = []
    for par in initial_parameters:
        if isinstance(par, numbers.Number):
            parsed_initial_pars.append(par)
        elif isinstance(par, str):
            # strings can be used to indicated special values. Only some values
            # are accepted (mostly just `halftime`)
            if par == 'halftime':
                parsed_initial_pars.append(tf / 2.)
            else:
                raise ValueError('"{}" is not a valid specifier'.format(par))
        elif (isinstance(par, (list, tuple)) and
              all(isinstance(par_elem, numbers.Number) for par_elem in par)):
            min_, max_ = par
            random_value = np.random.rand() * (max_ - min_) + min_
            parsed_initial_pars.append(random_value)
        else:  # otherwise we assume a callable that is a function of tf
            parsed_initial_pars.append(par(tf))
    return parsed_initial_pars


def _optimize_model_parameters_scan_times(
        times_to_try=None,
        hamiltonian=None,
        initial_state=None, target_state=None,
        initial_parameters=None,
        optimization_method=None,
        optimization_options=None, solver_options=None,
        stopping_condition=None):
    """Run a series of OC optimizations for different times.

    Parameters
    ----------
    times_to_try : list of positive numbers
        For each time in this list the optimization over the parameter will
        be performed (unless a stopping condition is also given, in which case
        not all of the provided time are necessary used).
    hamiltonian : TimeDependentHamiltonian instance
    initial_parameters : list
        List of the same length of the number of parameters defining the
        protocol (total time excluded).
        Each element is either a number or a pair of two numbers. If a number,
        this value is used for all optimizations as initial value for the
        corresponding parameter.
        If a pair of numbers, at every iteration a random value in the given
        range is used as initial value for the corresponding parameter.
    initial_state : qutip.Qobj
    target_state : qutip.Qobj
    optimization_method : string
    stopping_condition : float
        If None, all of the given times are scanned and the parameters are
        optimized for each one. If a single float integer, it is used as a
        criterion for stopping the scan before all times are checked: the
        functions exits when the optimization find a fidelity bigger than or
        equal to the given value.
    """
    protocol = hamiltonian.td_protocol
    # the +1 is for the fidelity, that is added to all the other parameters
    num_columns_result = protocol.num_parameters_to_save + 1
    pars_cols_names = list(protocol.hyperpars.keys()) + protocol.pars_names
    pars_cols_names = ['fid'] + pars_cols_names

    results = np.zeros(shape=[len(times_to_try), num_columns_result])
    for idx, tf in enumerate(times_to_try):
        logging.info('---- Iteration {}/{}'.format(idx + 1, len(times_to_try)))
        # parse initial parameters (we do this here to generate new random
        # values at each iteration)
        initial_pars = _parse_initial_parameters(initial_parameters, tf)

        try:
            result = optimize_model_parameters(
                hamiltonian=hamiltonian, initial_state=initial_state,
                target_state=target_state, tlist=tf,
                initial_parameters=initial_pars,
                optimization_method=optimization_method,
                optimization_options=optimization_options,
                solver_options=solver_options
            )
            fid = (1 - result.fun)**2  # the cost, result.fun, is 1 - sqrt(fid)
            if not 0 <= fid <= 1.01:
                logging.info('Optimization failed: got a nonphysical fidelity.')
            logging.info('Fidelity: {}'.format(fid))

            pars_to_save = list(protocol.hyperpars.values()) + list(result.x)
            results[idx] = [fid] + pars_to_save

            if stopping_condition is not None and fid >= stopping_condition:
                results = results[:idx + 1]
                logging.info('Stopping condition (fid >= {}) reached, '
                             'stopping.'.format(stopping_condition))
                break
        except KeyboardInterrupt:
            logging.info('Optimization interrupted, attempting to save '
                         'partial results.')
            results = results[:idx + 1]
            break

    results = pd.DataFrame(results, columns=pars_cols_names)
    return results


def find_best_protocol(
        problem_specification, optimization_specs,
        other_options={}
    ):
    """Higher-level interface to optimize_model_parameters.
    
    Parameters
    ----------
    problem_specification : dict
        Mandatory keys: 'model', 'model_parameters', 'task', 'time'.
        The accepted values for the `model` key are
            - 'rabi'
            - 'lmg'

        If model='rabi', the accepted values for `model_parameters` are
            - 'N'
            - 'Omega'
            - 'omega_0'
        If model='lmg', the accepted values for `model_parameters` are
            - 'num_spins'
        
        The accepted values for `task` are:
            - 'critical point state generation'

        The value of 'time' should be a positive number, representing the
        evolution time.

    optimization_specs : dict
        Accepted keys:
            - 'protocol'
            - 'protocol_options'
            - 'optimization_method'
            - 'optimization_options'
            - 'solver_options'
            - 'initial_parameters'
            - 'parameters_constraints'

        The accepted values for `protocol` are
            - 'doublebang'
            - 'bangramp'
            - 'crab'

        The accepted values for 'protocol_options' depend on 'protocol'.
        If protocol='crab' then the accepted values are
            - 'num_frequencies'

        The accepted values for 'optimization_method' are those accepted by
        scipy.optim.minimize, and similarly the value of 'optimization_options'
        is passed over to this function.
        Other values accepted for optimization_specs are:
            - 'parameters_constraints'

        The value of 'initial_parameters' can be either a numpy array with
        explicit values, or a string.

        The value of `parameters_constraints` can a list of pairs, with
        each pair specifying the constraints for one of the optimisation pars.
        If it instead a single list of two elements, it is instead used to
        put constraints on the overall height of the protocol (so not directly
        on the parameters that define it). This is notably necessary for the
        CRAB protocol.

    other_options : dict
        Accepted keys:
            - 'scan_times'
            - 'stopping_condition'

    """
    model = problem_specification['model']
    model_parameters = problem_specification['model_parameters']
    task = problem_specification['task']
    protocol_name = optimization_specs['protocol']
    optim_method = optimization_specs['optimization_method']
    optim_options = optimization_specs.get('optimization_options')
    solver_options = optimization_specs.get('solver_options')

    initial_state = None
    target_state = None

    if model == 'rabi':
        hamiltonian = rabi_model.RabiModel(
            N=model_parameters['N'],
            Omega=model_parameters['Omega'],
            omega_0=model_parameters['omega_0']
        )
    elif model == 'lmg':
        hamiltonian = lmg_model.LMGModel(
            num_spins=model_parameters['num_spins']
        )
    elif model == 'lz':
        hamiltonian = lz_model.LZModel(omega_0=model_parameters['omega_0'])
    else:
        raise ValueError("{} isn't a valid value for the model".format(model))

    if isinstance(task, str) and task == 'critical point':
        task = dict(initial_intensity=0,
                    final_intensity=hamiltonian.critical_value)
    elif not isinstance(task, dict):
        raise ValueError('`task` must be a dictionary or a string.')

    initial_state = hamiltonian.ground_state(task['initial_intensity'])
    target_state = hamiltonian.ground_state(task['final_intensity'])
    # determine protocol ansatz to use and parse options if needed
    # if `protocol_name` is NOT a string, then it is assumed to have been given
    # directly as a protocol_ansatz object
    if not isinstance(protocol_name, str):
        logging.info('Using custom protocol ansatz.')
        protocol = protocol_name
    elif protocol_name == 'doublebang':
        logging.info('Using doublebang protocol ansatz.')
        protocol = DoubleBangProtocolAnsatz()
    elif protocol_name == 'bangramp':
        logging.info('Using bangramp protocol ansatz.')
        protocol = BangRampProtocolAnsatz()
    elif protocol_name == 'crab':
        logging.info('Using CRAB protocol ansatz.')

        protocol_options = optimization_specs.get('protocol_options', {})
        # determine number of frequencies to use with CRAB protocol
        if 'num_frequencies' not in protocol_options:
            # default value kind of picked at random
            num_frequencies = 2
            logging.info('(CRAB) Default value of {} frequencies chosen for th'
                         'e optimization'.format(num_frequencies))
        else:
            num_frequencies = protocol_options['num_frequencies']
        logging.info('(CRAB) Using {} frequencies.'.format(num_frequencies))

        # fix starting and final points for pulse protocol in the case of
        # critical state generation task
        protocol = CRABProtocolAnsatz(num_frequencies=num_frequencies)

        protocol.fill_hyperpar_value(y0=task['initial_intensity'],
                                     y1=task['final_intensity'])
    else:
        raise ValueError('Unrecognised protocol.')

    # the scan_times option indicates that we want to perform a series of
    # optimizations for various values of the time parameter
    if 'scan_times' in other_options:

        # parse initial parameters
        critical_value = hamiltonian.critical_value
        if 'initial_parameters' not in optimization_specs:
            # if not explicitly given, use as default double the critical
            # point for intensities and the timeframe for times
            if protocol == 'doublebang':
                init_pars = [[0., 2 * critical_value], 'halftime',
                             [0., 2 * critical_value]]
            elif protocol == 'bangramp':
                init_pars = [[0., 2 * critical_value], 'halftime',
                             [0., 2 * critical_value],
                             [0., 2 * critical_value]]
            elif protocol == 'crab' or protocol == 'crabVarEndpoints':
                # I don't know, let's just try with amplitudes randomly
                # sampled in the [-1, 1] interval (why not right?)
                init_pars = [[-0.1, 0.1]] * (2 * protocol.num_frequencies)
            else:
                raise ValueError('For custom protocols the initial parameters '
                                 'must be given explicitly.')
        else:
            # if explicitly given, just assume the values make sense for
            # _optimize_model_parameters_scan_times
            init_pars = optimization_specs['initial_parameters']

        CRAB_AMPS_BOUNDS = [-200, 200]
        # parse parameters constraints
        if 'parameters_constraints' not in optimization_specs:
            # if not explicitly given, we generate boundaries similar to
            # those generated for default initial parameters
            critical_range = [-2 * critical_value, 2 * critical_value]
            if protocol == 'doublebang' or protocol == 'bangramp':
                protocol.constrain_intensities(critical_range)
            elif protocol == 'crab' or protocol == 'crabVarEndpoints':
                # not sure if this will work well in general, but in this
                # case we impose very weak constraints on the parameters,
                # and then the actual constraints are on the overall pulse
                protocol.constrain_all_amplitudes(CRAB_AMPS_BOUNDS)
                protocol.set_total_height_constraints(critical_range)
        else:
            pars_constraints = optimization_specs['parameters_constraints']
            if isinstance(pars_constraints, (tuple, list)):
                if len(pars_constraints) == 2:
                    if protocol == 'crab' or protocol == 'crabVarEndpoints':
                        protocol.constrain_all_amplitudes(CRAB_AMPS_BOUNDS)
                        protocol.set_total_height_constraints(pars_constraints)
                    else:
                        protocol.constrain_intensities(pars_constraints)
            else:
                protocol.add_parameter_constraints(pars_constraints)

        logging.info('Using parameters constraints: {}'.format(
            protocol.pars_constraints))
        logging.info('Using total height constraint: {}'.format(
            protocol.total_height_constraint))

        hamiltonian.td_protocol = protocol
        # run optimization
        stopping_condition = other_options.get('stopping_condition', None)
        results = _optimize_model_parameters_scan_times(
            times_to_try=other_options['scan_times'],
            hamiltonian=hamiltonian,
            initial_state=initial_state, target_state=target_state,
            initial_parameters=init_pars,
            optimization_method=optim_method,
            optimization_options=optim_options,
            solver_options=solver_options,
            stopping_condition=stopping_condition
        )
        return results
    
    else:  # only scan_times works atm
        raise NotImplementedError('Only scan_times works atm, sorry.')
    

