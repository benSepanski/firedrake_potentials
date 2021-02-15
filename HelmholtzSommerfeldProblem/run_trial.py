import os
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

from os.path import isfile, join
from time import sleep

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
from firedrake import sqrt, Constant, pi, exp, SpatialCoordinate, \
        trisurf, warning, product, real, conditional, norms, Mesh

from methods import run_method

from firedrake.petsc import OptionsManager, PETSc
from firedrake.solving_utils import KSPReasons
from utils.norm_functions import l2_norm, h1_norm
from utils.to_2nd_order import to_2nd_order

import faulthandler
faulthandler.enable()

logger = logging.getLogger("HelmholtzTrial")
handler = logging.StreamHandler()
# format string from pytential/pytential/log.py
c_format = logging.Formatter(
        "[%(name)s][%(levelname)s]  %(message)s (%(filename)s:%(lineno)d)"
        )
handler.setFormatter(c_format)
logger.addHandler(handler)
logger.setLevel(level=logging.INFO)

# {{{ Trial settings for user to modify

mesh_options = {
        # Must be one of the keys of mesh_options['mesh_options']
        'mesh_name': 'circle_in_square',
        # clmax of coarsest mesh
        'element_size': 2**-1,
        # number of refinements
        'num_refinements': 6,
    # mesh-specific options
    'mesh_options': {
        'circle_in_square_no_pml': {
            'radius': 1.0,
             # This can be a list of floats or just one float
            'outer_side_length': [2.25, 2.5, 3.0, 4.0, 5.0, 6.0],
        },
        'circle_in_square': {
            'radius': 1.0,
             # This can be a list of floats or just one float
            'outer_side_length': 6.0,
        },
        'ball_in_cube': {
            'radius': 1.0,
             # This can be a list of floats or just one float
            'outer_side_length': 6.0,
        },
        'annulus': {
            'inner_radius': 1.0,
            'outer_radius': 2.0,
        },
        'betterplane': {},
    },
}

#kappa_list = [10.0]
kappa_list = [0.1, 1.0, 5.0, 10.0]
#degree_list = [1]
#degree_list = [2, 3]
degree_list = [4]
method_list = ['transmission', 'pml', 'nonlocal']
# to use pyamg for the nonlocal method, use 'pc_type': 'pyamg'
# SPECIAL KEYS for preconditioning (these are all passed through petsc options
#              via the command line or *method_to_kwargs*):
# 'pyamg_maxiter' and 'pyamg_tol' to change the default pyamg maxiter or tol
# for preconditioning
#
# Use 'gamma' or 'beta' for an altering of the preconditioner (non-pyamg).
"""
        'solver_parameters': {'pc_type': 'pyamg',
                              'ksp_type': 'fgmres',
                              'ksp_max_it': 50,
                              'pyamg_tol': 1e-50,
                              'pyamg_maxiter': 3,
                              'ksp_monitor_true_residual': None,
                              },
"""
method_to_kwargs = {
    'transmission': {
        'options_prefix': 'transmission',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_type': 'preonly'
                              }
    },
    'pml': {
        'pml_type': 'bdy_integral',
        'options_prefix': 'pml',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_type': 'preonly'
                              }
    },
    'nonlocal': {
        'options_prefix': 'nonlocal',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_monitor': None,
                              }
    }
}
"""
'solver_parameters': {'pc_type': 'pyamg',
                      'ksp_type': 'fgmres',
                      'ksp_max_it': 50,
                      'pyamg_tol': 1e-50,
                      'pyamg_maxiter': 3,
                      'ksp_monitor_true_residual': None,
                      },
"""

# Use cache if have it?
use_cache = False  # pylint: disable=C0103

# Num refinements?

# Visualize solutions?
visualize = False  # pylint: disable=C0103


def get_fmm_order_or_tol(kappa, h):
    """
    Set the fmm order or tol for each (kappa, h) pair

    :arg kappa: The wave number
    :arg h: The maximum characteristic length of the mesh
    :return: A dict d with keys 'fmm_order' and 'fmm_tol'.
             Exactly one of d['fmm_order'] or d['fmm_tol']
             is *None*
    """
    from math import log
    # FMM order to get tol accuracy
    fmm_order_and_tol = {}
    fmm_tol = 1e-31
    global c
    if mesh_dim == 2:
        c = 0.5  # pylint: disable=C0103
    elif mesh_dim == 3:
        c = 0.75  # pylint: disable=C0103
    fmm_order = int(log(fmm_tol, c)) - 1

    # Exactly one of these must be None
    return {'fmm_order': None, 'fmm_tol': fmm_tol}
    #return {'fmm_order': fmm_order, 'fmm_tol': None}
    #return {'fmm_order': False, 'fmm_tol': None}

# }}}


# Extract mesh options
mesh_name = mesh_options['mesh_name']  # pylint: disable=C0103
element_size = mesh_options['element_size']  # pylint: disable=C0103
num_refinements = mesh_options['num_refinements']  # pylint: disable=C0103
try:
    mesh_options = mesh_options['mesh_options'][mesh_name]
except KeyError:
    raise KeyError("Unrecognized mesh name {mesh_name}. "
                   "Mesh name must be one of {meshes}".format(
                       mesh_name=mesh_name,
                       meshes=mesh_options['mesh_options'].keys()))


# Open cache file to get any previously computed results
cache_file_name = "data/" + mesh_name + '.csv'

def read_cache():
    """
    Read cache from the cache file.
    Returns the cacche
    """
    # If no cache file, return empty dict
    if not isfile(cache_file_name):
        return {}
    # Repeatedly try to open and write to the cache file
    while True:
        try:
            in_file = open(cache_file_name)
            cache_reader = csv.DictReader(in_file)
            cache = {}

            # Each entry is the set of all input (key, value) pairs
            # and all output (key, value) pairs, so separate those
            # into input_ and output
            for i, entry in enumerate(cache_reader):

                output = {}
                input_ = dict(entry)
                for output_name in ['L2 Error', 'H1 Error', 'ndofs',
                                    'L2 True Norm', 'H1 True Norm',
                                    'Iteration Number', 'Residual Norm',
                                    'Converged Reason',
                                    'Min Extreme Singular Value',
                                    'Max Extreme Singular Value']:
                    output[output_name] = entry[output_name]
                    del input_[output_name]
                # Cache maps input (k, v) -> output (k, v)
                cache[frozenset(input_.items())] = output

            in_file.close()
            return cache

        except (OSError, IOError):
            sleep(1)


logger.info("Reading cache...")
cache = read_cache()
logger.info("Cache read in")

mesh_dim = None  # pylint: disable=C0103
if mesh_name in ['annulus', 'circle_in_square', 'circle_in_square_no_pml']:
    mesh_dim = 2  # pylint: disable=C0103
    hankel_cutoff = 80  # pylint: disable=C0103

    if mesh_name == 'circle_in_square':
        inner_bdy_id = 3  # pylint: disable=C0103
        outer_bdy_id = 1
        non_sponge_region = 4
        pml_min = [2, 2]
        pml_max = None  # Set during iteration based on outer side length
        # if only one outer side length given, convert to iterable
        if not isinstance(mesh_options['outer_side_length'], list):
            mesh_options['outer_side_length'] = [mesh_options['outer_side_length']]
        for outer_side_length in mesh_options['outer_side_length']:
            if not isinstance(outer_side_length, float):
                raise TypeError("Each outer side length must be a float")
        mesh_file_names = ["circle_in_square-rad{rad}-side{side}"
                           .format(rad=mesh_options['radius'], side=side)
                           for side in mesh_options['outer_side_length']]
    elif mesh_name == 'circle_in_square_no_pml':
        inner_bdy_id = 2  # pylint: disable=C0103
        outer_bdy_id = 1
        non_sponge_region = 3
        if 'pml' in method_list:
            raise ValueError("pml not supported on 'circle_in_square_no_pml' mesh")
        pml_min = None
        pml_max = None  # Set during iteration based on outer side
        # if only one outer side length given, convert to iterable
        if not isinstance(mesh_options['outer_side_length'], list):
            mesh_options['outer_side_length'] = [mesh_options['outer_side_length']]
        for outer_side_length in mesh_options['outer_side_length']:
            if not isinstance(outer_side_length, float):
                raise TypeError("Each outer_side_length must be a float")
        mesh_file_names = ["circle_in_square_no_pml-rad{rad}-side{side}"
                           .format(rad=mesh_options['radius'], side=side)
                           for side in mesh_options['outer_side_length']]
    elif mesh_name == 'annulus':
        inner_bdy_id = 2  # pylint: disable=C0103
        outer_bdy_id = 1  # pylint: disable=C0103
        non_sponge_region = None
        if 'pml' in method_list:
            raise ValueError('pml not supported on annulus mesh')
        pml_min = None  # pylint: disable=C0103
        pml_max = None  # pylint: disable=C0103
        mesh_file_names = [
            "annulus-inner_rad{inner_rad}-outer_rad{outer_rad}"
            .format(inner_rad=mesh_options['inner_radius'],
                    outer_rad=mesh_options['outer_radius'])
            ]
        raise NotImplementedError("annulus bdy ids are incorrect")

elif mesh_name in ['ball_in_cube', 'betterplane']:
    mesh_dim = 3  # pylint: disable=C0103
    hankel_cutoff = 50  # pylint: disable=C0103

    if mesh_name == 'ball_in_cube':
        inner_bdy_id = 3  # pylint: disable=C0103
        outer_bdy_id = 1
        non_sponge_region = 5
        pml_min = [2, 2, 2]
        pml_max = None  # Set during iteration based on outer_side_length
        # if only one outer_side_length given, convert to iterable
        if not isinstance(mesh_options['outer_side_length'], list):
            mesh_options['outer_side_length'] = [mesh_options['outer_side_length']]
        for outer_side_length in mesh_options['outer_side_length']:
            if not isinstance(outer_side_length, float):
                raise TypeError("Each outer_side_length must be a float")
        mesh_file_names = ["ball_in_cube-rad{rad}-side{side}"
                           .format(rad=mesh_options['radius'], side=side)
                           for side in mesh_options['outer_side_length']]

    elif mesh_name == 'betterplane':
        inner_bdy_id = list(range(7, 32))  # pylint: disable=C0103
        outer_bdy_id = [1, 2, 3, 4, 5, 6]  # pylint: disable=C0103
        non_sponge_region = None
        pml_min = [11, 4.62, 10.5]
        pml_max = [12, 5.62, 11.5]
        mesh_file_names = "betterplane"
        raise NotImplementedError("betterplane requires a source layer"
                                  " involving multiple boundary tags. "
                                  " Not yet implemented")

else:
    raise ValueError("Unrecognized mesh name '%s'." % mesh_name)


# Set kwargs that don't expect user to change
# (some of these are for just pml, but we don't
#  expect the user to want to change them
#
# The default solver parameters here are the defaults for
# a :class:`LinearVariationalSolver`, see
# https://www.firedrakeproject.org/solving-interface.html#id19
global_kwargs = {'scatterer_bdy_id': inner_bdy_id,
                 'outer_bdy_id': outer_bdy_id,
                 'pml_min': pml_min,
                 'pml_max': pml_max,
                 'solver_parameters': {'snes_type': 'ksponly',
                                       'ksp_type': 'gmres',
                                       'ksp_gmres_restart': 30,
                                       'ksp_rtol': 1.0e-12,
                                       'ksp_atol': 1.0e-50,
                                       'ksp_divtol': 1e4,
                                       'ksp_max_it': 10000,
                                       'pc_type': 'ilu',
                                       'pc_side': 'left',
                                       },
                 }

# Ready kwargs by defaulting any absent kwargs to the global ones
for mkey in method_to_kwargs:
    for gkey in global_kwargs:
        if gkey not in method_to_kwargs[mkey]:
            method_to_kwargs[mkey][gkey] = global_kwargs[gkey]

order = 2 if np.any(np.array(degree_list) > 1) else 1  # pylint: disable=C0103

# may have multiple mesh files if multiple outer_side_lengths
current_mesh_file_name = []
cell_sizes = []
outer_side_lengths = []
for i, mesh_file_name in enumerate(mesh_file_names):
    cell_sizes += [element_size * 2**-i for i in range(num_refinements+1)]
    current_mesh_file_name += [mesh_file_name + "-h%.5e.msh" % cell_size for cell_size in cell_sizes[-num_refinements-1:]]
    outer_side_length = ''
    if 'outer_side_length' in mesh_options:
        outer_side_length = mesh_options['outer_side_length'][i]
    outer_side_lengths += [outer_side_length for _ in range(num_refinements + 1)]

# Verify mesh names exist
for  mesh_file_name in current_mesh_file_name:
    if not isfile(join('meshes', mesh_file_name)):
        raise ValueError(
            "{mesh_file} not found. Modify bin/make_meshes to "
            "generate appropriate mesh, or modify mesh options."
            .format(mesh_file=join('meshes', mesh_file_name)))

# {{{ Get setup options for each method
for method in method_list:
    # Get the solver parameters
    solver_parameters = dict(global_kwargs.get('solver_parameters', {}))
    for k, v in method_to_kwargs[method].get('solver_parameters', {}).items():
        solver_parameters[k] = v

    options_prefix = method_to_kwargs[method].get('options_prefix', None)

    options_manager = OptionsManager(solver_parameters, options_prefix)
    options_manager.inserted_options()
    method_to_kwargs[method]['solver_parameters'] = options_manager.parameters
# }}}


# Store error and functions
results = {}

iteration = 0  # pylint: disable=C0103
total_iter = len(current_mesh_file_name) * len(degree_list) * \
             len(kappa_list) * len(method_list)

field_names = ('Mesh Order', 'Outer Side Length',
               'h', 'degree', 'kappa', 'method',
               'pc_type', 'pc_side', 'FMM Order', 'FMM Tol', 'ndofs',
               'L2 Error', 'H1 Error', 'L2 True Norm',
               'H1 True Norm', 'Iteration Number',
               'gamma', 'beta', 'ksp_type',
               'Residual Norm', 'Converged Reason', 'ksp_rtol', 'ksp_atol',
               'Min Extreme Singular Value', 'Max Extreme Singular Value',
               'pyamg_maxiter', 'pyamg_tol')

setup_info = {'Mesh Order': str(order)}
if mesh_name in ['circle_in_square', 'circle_in_square_no_pml', 'ball_in_cube']:
    setup_info['Outer Side Length'] = str(mesh_options['outer_side_length'])
else:
    setup_info['Outer Side Length'] = str('')

for mesh_file_name, cell_size, outer_side_length in zip(current_mesh_file_name,
                                                        cell_sizes,
                                                        outer_side_lengths):
    setup_info['h'] = str(cell_size)
    setup_info['Outer Side Length'] = str(outer_side_length)
    if(outer_side_length != '' and 'pml' in method_to_kwargs):
        method_to_kwargs['pml']['pml_max'] = [outer_side_length / 2 for _ in range(mesh_dim)]

    mesh = None

    for degree in degree_list:
        setup_info['degree'] = str(degree)

        # The first time we run with a new mesh+fspace degree,
        # clear any memoized objects
        clear_memoized_objects = True

        for kappa in kappa_list:
            if isinstance(kappa, int):
                setup_info['kappa'] = str(float(kappa))
            else:
                setup_info['kappa'] = str(kappa)
            true_sol = None  # pylint: disable=C0103

            trial = {'degree': degree,
                     'true_sol': true_sol}

            for method in method_list:
                solver_params = method_to_kwargs[method]['solver_parameters']

                setup_info['method'] = str(method)
                setup_info['pc_type'] = str(solver_params['pc_type'])
                setup_info['pc_side'] = str(solver_params['pc_side'])
                setup_info['ksp_type'] = str(solver_params['ksp_type'])
                if solver_params['ksp_type'] == 'preonly':
                    setup_info['ksp_rtol'] = ''
                    setup_info['ksp_atol'] = ''
                else:
                    setup_info['ksp_rtol'] = str(solver_params['ksp_rtol'])
                    setup_info['ksp_atol'] = str(solver_params['ksp_atol'])

                if method == 'nonlocal':
                    # Get fmm order or fmm tol
                    fmm_order_or_tol = get_fmm_order_or_tol(kappa, cell_size)
                    fmm_order = fmm_order_or_tol['fmm_order']
                    fmm_tol = fmm_order_or_tol['fmm_tol']
                    # Store fmm order/fmm tol in setup info
                    setup_info['FMM Order'] = '' if fmm_order is None else str(fmm_order)
                    setup_info['FMM Tol'] = '' if fmm_tol is None else str(fmm_tol)
                    # Put fmm order/tol into kwargs for the method
                    method_to_kwargs[method]['FMM Order'] = fmm_order
                    method_to_kwargs[method]['FMM Tol'] = fmm_tol 
                else:
                    setup_info['FMM Order'] = ''
                    setup_info['FMM Tol'] = ''

                # Add gamma/beta & pyamg info to setup_info if there, else make
                # sure it's recorded as absent in special_key
                if solver_params['pc_type'] != 'pyamg':
                    solver_params.pop('pyamg_maxiter', None)
                    solver_params.pop('pyamg_tol', None)
                for special_key in ['gamma',
                                    'beta',
                                    'pyamg_maxiter',
                                    'pyamg_tol']:
                    if special_key in solver_params:
                        # Make sure gets converted from float
                        if special_key == 'pyamg_tol':
                            setup_info[special_key] = \
                                float(solver_params[special_key])
                        setup_info[special_key] = \
                            str(solver_params[special_key])
                    else:
                        setup_info[special_key] = ''

                    # Overwrite other options if used with options prefix
                    options_prefix = \
                        method_to_kwargs[method].get('options_prefix', None)
                    if options_prefix:
                        if options_prefix[-1] != '_':
                            options_prefix += '_'
                        new_special_key = options_prefix + special_key
                        # FIXME: CHECK FROM COMMAND LINE
                        if new_special_key in solver_params:
                            setup_info[special_key] = \
                                str(solver_params[new_special_key])

                # Gets computed solution, prints and caches
                key = frozenset(setup_info.items())

                new_result = not use_cache or key not in cache
                if new_result:
                    # Build mesh if not done so already
                    if mesh is None:
                        logger.info("Reading Mesh %s with element size %s " +
                                    " and mesh order %s...",
                                    mesh_file_name,
                                    cell_size,
                                    order)
                        # read in using open-cascade with 0 refinements, this allows
                        # for order 2 meshes
                        mesh = Mesh(join('meshes/', mesh_file_name))
                        if order == 2:
                            if mesh_name not in ['circle_in_square', 'ball_in_cube', 'circle_in_square_no_pml']:
                                raise NotImplementedError("2nd order mesh only avaiilbale" +
                                        " for circle_in_square, circle_in_square_no_pml, or ball_in_cube. " +
                                        " Not available for '%s'" % mesh_name)
                            mesh = to_2nd_order(mesh, inner_bdy_id, mesh_options['radius'])
                        logger.info("Mesh read in")

                        if visualize:
                            from firedrake import triplot
                            import matplotlib.pyplot as plt
                            triplot(mesh)
                            plt.title("h=%.2e" % cell_size)
                            plt.show()

                    # make sure to store mesh in trial
                    trial['mesh'] = mesh

                    kwargs = method_to_kwargs[method]
                    true_sol, comp_sol, snes_or_ksp = run_method.run_method(
                        trial, method, kappa,
                        cl_ctx=cl_ctx,
                        queue=queue,
                        clear_memoized_objects=clear_memoized_objects,
                        comp_sol_name=method + " Computed Solution", **kwargs)
                    # After we've started running on this mesh+fspace degree, don't
                    # clear memoized objects until we get to the next mesh+fspace degree
                    clear_memoized_objects = False

                    if isinstance(snes_or_ksp, PETSc.SNES):
                        ksp = snes_or_ksp.getKSP()
                    elif isinstance(snes_or_ksp, PETSc.KSP):
                        ksp = snes_or_ksp
                    else:
                        raise ValueError("snes_or_ksp must be of type"
                                         "PETSc.SNES or PETSc.KSP")

                    cache[key] = {}

                    diff = true_sol - comp_sol
                    l2_err = abs(l2_norm(diff, region=non_sponge_region))
                    h1_err = abs(h1_norm(diff, region=non_sponge_region))
                    l2_true_norm = abs(l2_norm(true_sol, region=non_sponge_region))
                    h1_true_norm = abs(h1_norm(true_sol, region=non_sponge_region))

                    cache[key]['L2 Error'] = l2_err
                    cache[key]['H1 Error'] = h1_err
                    cache[key]['L2 True Norm'] = l2_true_norm
                    cache[key]['H1 True Norm'] = h1_true_norm

                    ndofs = true_sol.dat.data.shape[0]
                    cache[key]['ndofs'] = str(ndofs)
                    # Grab iteration number if not preonly
                    if solver_params['ksp_type'] != 'preonly':
                        cache[key]['Iteration Number'] = \
                            ksp.getIterationNumber()
                    # Get residual norm and converged reason
                    cache[key]['Residual Norm'] = \
                        ksp.getResidualNorm()
                    cache[key]['Converged Reason'] = \
                        KSPReasons[ksp.getConvergedReason()]

                    # If using gmres, estimate extreme singular values
                    compute_sing_val_params = set([
                        'ksp_compute_singularvalues',
                        'ksp_compute_eigenvalues',
                        'ksp_monitor_singular_value'])
                    if solver_params['ksp_type'] == 'gmres' and \
                            compute_sing_val_params & solver_params.keys():
                        emax, emin = ksp.computeExtremeSingularValues()
                        cache[key]['Min Extreme Singular Value'] = emin
                        cache[key]['Max Extreme Singular Value'] = emax

                    if visualize:
                        try:
                            trisurf(comp_sol)
                            trisurf(true_sol)
                            plt.show()
                        except Exception as e:
                            warning("Cannot plot figure. Error msg: '%s'", e)

                else:
                    ndofs = cache[key]['ndofs']
                    l2_err = cache[key]['L2 Error']
                    h1_err = cache[key]['H1 Error']

                iteration += 1
                print('iter:   %s / %s' % (iteration, total_iter))
                print('h:     ', cell_size)
                print("ndofs: ", ndofs)
                print("kappa: ", kappa)
                print("method:", method)
                print('degree:', degree)
                if setup_info['method'] == 'nonlocal':
                    if fmm_order is not None and fmm_order != False:
                        if mesh_dim == 2:
                            c = 0.5  # pylint: disable=C0103
                        else:
                            c = 0.75  # pylint: disable=C0103
                        print('Epsilon= %.2f^(%d+1) = %e'
                              % (c, fmm_order, c**(fmm_order+1)))
                    elif fmm_order != False:
                        print("FMM Tol=%.2e" % fmm_tol)
                if len(outer_side_lengths) > 1:
                    print("Outer Side Length:", outer_side_length)

                print("L2 Err: ", l2_err)
                print("H1 Err: ", h1_err)
                print()

                # write to cache if necessary (after every computation)
                if new_result:
                    logger.info("Writing to cache...")

                    ### TRY TO KEEP CACHE ERRORS LOW. MULTIPLE PROCESSES
                    ### CAN RUN AT THE SAME TIME, BUT IT IS POSSIBLE THAT
                    ### THIS WILL RESULT IN SOME DATA BEING LOST.

                    # read cache off disk
                    cache_on_disk = read_cache()

                    # Update our cache if it's missing anything
                    for key in cache_on_disk:
                        if key not in cache:
                            cache[key] = cache_on_disk[key]

                    # Write our cache to disk.
                    #
                    # First write to a temp file, that way
                    # if we hit I/O errors we don't lose a bunch of data
                    for out_file_name in [cache_file_name + '.swp', cache_file_name]:
                        out_file = open(out_file_name, 'w')

                        cache_writer = csv.DictWriter(out_file, field_names)
                        cache_writer.writeheader()

                        # {{{ Re-write all data

                        for key in cache:
                            row = dict(key)
                            for output in cache[key]:
                                row[output] = cache[key][output]
                            cache_writer.writerow(row)

                        # }}}

                        out_file.close()

                    logger.info("cache closed")
    del mesh
