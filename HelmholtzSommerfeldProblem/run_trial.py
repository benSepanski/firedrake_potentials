import os
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

from os.path import isfile, join

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
from firedrake import sqrt, Constant, pi, exp, SpatialCoordinate, \
        trisurf, warning, product, real, conditional, norms

from methods import run_method

from firedrake.petsc import OptionsManager, PETSc
from firedrake.solving_utils import KSPReasons
from utils.hankel_function import hankel_function

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
        'num_refinements': 4,
    # mesh-specific options
    'mesh_options': {
        'circle_in_square': {
            'radius': 1.0,
             # This can be a list of floats or just one float
            'cutoff_size': 1.0,
        },
        'ball_in_cube': {
            'radius': 1.0,
             # This can be a list of floats or just one float
            'cutoff_size': 1.0,
        },
        'annulus': {
            'inner_radius': 1.0,
            'outer_radius': 2.0,
        },
        'betterplane': {},
    },
}

#kappa_list = [0.1, 1.0, 5.0, 10.0]
kappa_list = [0.1]
degree_list = [1]
#method_list = ['nonlocal', 'pml', 'transmission']
method_list = ['nonlocal']
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
        'queue': queue,
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
use_cache = True  # pylint: disable=C0103

# Write over duplicate trials?
write_over_duplicate_trials = False  # pylint: disable=C0103

# Num refinements?

# Visualize solutions?
visualize = False  # pylint: disable=C0103


def get_fmm_order(kappa, h):
    """
    Set the fmm order for each (kappa, h) pair

    :arg kappa: The wave number
    :arg h: The maximum characteristic length of the mesh
    """
    from math import log
    # FMM order to get tol accuracy
    tol = 1e-4
    global c
    if mesh_dim == 2:
        c = 0.5  # pylint: disable=C0103
    elif mesh_dim == 3:
        c = 0.75  # pylint: disable=C0103
    return int(log(tol, c)) - 1

# }}}


# Extract mesh options
mesh_name = mesh_options['mesh_name']  # pylint: disable=C0103
element_size = mesh_options['element_size']  # pylint: disable=C0103
num_refinements = mesh_options['num_refinements']  # pylint: disable=C0103
if num_refinements > 3:
    raise ValueError("May produce incorrect results when num refinements is high.")
try:
    mesh_options = mesh_options['mesh_options'][mesh_name]
except KeyError:
    raise KeyError("Unrecognized mesh name {mesh_name}. "
                   "Mesh name must be one of {meshes}".format(
                       mesh_name=mesh_name,
                       meshes=mesh_options['mesh_options'].keys()))

# Open cache file to get any previously computed results
logger.info("Reading cache...")
cache_file_name = "data/" + mesh_name + '.csv'
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
except (OSError, IOError):
    cache = {}
logger.info("Cache read in")

uncached_results = {}

if write_over_duplicate_trials:
    uncached_results = cache

mesh_dim = None  # pylint: disable=C0103
if mesh_name in ['annulus', 'circle_in_square']:
    mesh_dim = 2  # pylint: disable=C0103
    hankel_cutoff = 80  # pylint: disable=C0103

    if mesh_name == 'circle_in_square':
        inner_bdy_id = 5  # pylint: disable=C0103
        outer_bdy_id = [1, 2, 3, 4]
        pml_min = [2, 2]
        pml_max = None  # Set during iteration based on cutoff size 
        # if only one cutoff size given, convert to iterable
        if not isinstance(mesh_options['cutoff_size'], list):
            mesh_options['cutoff_size'] = [mesh_options['cutoff_size']]
        for cutoff_size in mesh_options['cutoff_size']:
            if not isinstance(cutoff_size, float):
                raise TypeError("Each cutoff size must be a float")
        mesh_file_names = ["circle_in_square-rad{rad}-side{side}.step"
                           .format(rad=mesh_options['radius'],
                                   side=2 * (2 + cutoff))
                           for cutoff in mesh_options['cutoff_size']]
    elif mesh_name == 'annulus':
        inner_bdy_id = 2  # pylint: disable=C0103
        outer_bdy_id = 1  # pylint: disable=C0103
        if 'pml' in method_list:
            raise ValueError('pml not supported on annulus mesh')
        pml_min = None  # pylint: disable=C0103
        pml_max = None  # pylint: disable=C0103
        mesh_file_names = [
            "annulus-inner_rad{inner_rad}-outer_rad{outer_rad}.step"
            .format(inner_rad=mesh_options['inner_radius'],
                    outer_rad=mesh_options['outer_radius'])
            ]

elif mesh_name in ['ball_in_cube', 'betterplane']:
    mesh_dim = 3  # pylint: disable=C0103
    hankel_cutoff = 50  # pylint: disable=C0103

    if mesh_name == 'ball_in_cube':
        inner_bdy_id = 7  # pylint: disable=C0103
        outer_bdy_id = [1, 2, 3, 4, 5, 6]
        pml_min = [2, 2, 2]
        pml_max = None  # Set during iteration based on cutoff size 
        # if only one cutoff size given, convert to iterable
        if not isinstance(mesh_options['cutoff_size'], list):
            mesh_options['cutoff_size'] = [mesh_options['cutoff_size']]
        for cutoff_size in mesh_options['cutoff_size']:
            if not isinstance(cutoff_size, float):
                raise TypeError("Each cutoff size must be a float")
        mesh_file_names = ["ball_in_cube-rad{rad}-side{side}.step"
                           .format(rad=mesh_options['radius'],
                                   side=2 * (2 + cutoff))
                           for cutoff in mesh_options['cutoff_size']]

    elif mesh_name == 'betterplane':
        inner_bdy_id = list(range(7, 32))  # pylint: disable=C0103
        outer_bdy_id = [1, 2, 3, 4, 5, 6]  # pylint: disable=C0103
        pml_min = [11, 4.62, 10.5]
        pml_max = [12, 5.62, 11.5]
        mesh_file_names = "betterplane.step"
        raise NotImplementedError("betterplane requires a source layer"
                                  " involving multiple boundary tags. "
                                  " Not yet implemented")

else:
    raise ValueError("Unrecognized mesh name '%s'." % mesh_name)

for  mesh_file_name in mesh_file_names:
    if not isfile(join('meshes', mesh_file_name)):
        raise ValueError(
            "{mesh_file} not found. Modify bin/make_meshes to "
            "generate appropriate mesh, or modify mesh options."
            .format(mesh_file=join('meshes', mesh_file_name)))


def get_true_sol_expr(spatial_coord):
    """
    Get the ufl expression for the true solution
    """
    if mesh_dim == 3:
        x, y, z = spatial_coord  # pylint: disable=C0103
        norm = sqrt(x**2 + y**2 + z**2)
        return Constant(1j / (4*pi)) / norm * exp(1j * kappa * norm)

    if mesh_dim == 2:
        x, y = spatial_coord  # pylint: disable=C0103
        return Constant(1j / 4) * hankel_function(kappa * sqrt(x**2 + y**2),
                                                  n=hankel_cutoff)
    raise ValueError("Only meshes of dimension 2, 3 supported")


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
                                       'ksp_rtol': 1.0e-7,
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

from firedrake import OpenCascadeMeshHierarchy
order = 2 if np.any(np.array(degree_list) > 1) else 1  # pylint: disable=C0103

# may have multiple mesh files if multiple cutoff sizes
meshes = []
cell_sizes = []
mesh_names = []
cutoff_sizes = []
for i, mesh_file_name in enumerate(mesh_file_names):
    logger.info("Building Mesh Hierarchy %s / %s with mesh order %s...",
                i+1,
                len(mesh_file_names),
                order)
    mesh_hierarchy = OpenCascadeMeshHierarchy(join('meshes/', mesh_file_name),
                                              element_size,
                                              num_refinements,
                                              order=order,
                                              project_refinements_to_cad=True,
                                              cache=False)

    meshes += mesh_hierarchy.meshes
    cell_sizes += [element_size * 2**-i for i in range(num_refinements+1)]
    mesh_names += [mesh_name + str(cell_size) for cell_size in cell_sizes[-num_refinements-1:]]
    cutoff_size = ''
    if 'cutoff_size' in mesh_options:
        cutoff_size = mesh_options['cutoff_size'][i]
    cutoff_sizes += [cutoff_size for _ in range(num_refinements + 1)]
    logger.info("Mesh Hierarchy prepared.")

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
total_iter = len(mesh_names) * len(degree_list) * \
             len(kappa_list) * len(method_list)

field_names = ('Mesh Order', 'Cutoff Size',
               'h', 'degree', 'kappa', 'method',
               'pc_type', 'pc_side', 'FMM Order', 'ndofs',
               'L2 Error', 'H1 Error', 'L2 True Norm',
               'H1 True Norm', 'Iteration Number',
               'gamma', 'beta', 'ksp_type',
               'Residual Norm', 'Converged Reason', 'ksp_rtol', 'ksp_atol',
               'Min Extreme Singular Value', 'Max Extreme Singular Value',
               'pyamg_maxiter', 'pyamg_tol')

setup_info = {'Mesh Order': str(order)}
if mesh_name in ['circle_in_square', 'ball_in_cube']:
    setup_info['Cutoff Size'] = str(mesh_options['cutoff_size'])
else:
    setup_info['Cutoff Size'] = str('')

for mesh, mesh_name, cell_size, cutoff_size in zip(meshes,
                                                   mesh_names,
                                                   cell_sizes,
                                                   cutoff_sizes):
    if visualize:
        from firedrake import triplot
        import matplotlib.pyplot as plt
        triplot(mesh)
        plt.title("h=%.2e" % cell_size)
        plt.show()

    setup_info['h'] = str(cell_size)
    setup_info['Cutoff Size'] = str(cutoff_size)
    if(cutoff_size != '' and 'pml' in method_to_kwargs):
        method_to_kwargs['pml']['pml_max'] = [2 + cutoff_size for _ in range(mesh_dim)]

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
            true_sol_expr = None  # pylint: disable=C0103

            trial = {'mesh': mesh,
                     'degree': degree,
                     'true_sol_expr': true_sol_expr}

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
                    fmm_order = get_fmm_order(kappa, cell_size)
                    setup_info['FMM Order'] = str(fmm_order)
                    method_to_kwargs[method]['FMM Order'] = fmm_order
                else:
                    setup_info['FMM Order'] = ''

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

                if not use_cache or key not in cache:
                    # {{{  Compute true solution expression if haven't already
                    if true_sol_expr is None:
                        true_sol_expr = \
                            get_true_sol_expr(SpatialCoordinate(mesh))
                        trial['true_sol_expr'] = true_sol_expr
                    # }}}

                    kwargs = method_to_kwargs[method]
                    true_sol, comp_sol, snes_or_ksp = run_method.run_method(
                        trial, method, kappa,
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

                    uncached_results[key] = {}

                    # 1 in inner-region, 0 in PML region
                    one_in_inner_region = product(
                        [conditional(abs(real(coord)) >= real(min_),
                                     Constant(0.0),
                                     Constant(1.0))
                         for coord, min_ in zip(SpatialCoordinate(mesh),
                                                pml_min)])
                    diff = true_sol - comp_sol
                    l2_err = abs(norms.norm(diff * one_in_inner_region, norm_type="L2"))
                    h1_err = abs(norms.norm(diff * one_in_inner_region, norm_type="H1"))
                    l2_true_norm = abs(norms.norm(true_sol, norm_type="L2"))
                    h1_true_norm = abs(norms.norm(true_sol, norm_type="H1"))

                    uncached_results[key]['L2 Error'] = l2_err
                    uncached_results[key]['H1 Error'] = h1_err
                    uncached_results[key]['L2 True Norm'] = l2_true_norm
                    uncached_results[key]['H1 True Norm'] = h1_true_norm

                    ndofs = true_sol.dat.data.shape[0]
                    uncached_results[key]['ndofs'] = str(ndofs)
                    # Grab iteration number if not preonly
                    if solver_params['ksp_type'] != 'preonly':
                        uncached_results[key]['Iteration Number'] = \
                            ksp.getIterationNumber()
                    # Get residual norm and converged reason
                    uncached_results[key]['Residual Norm'] = \
                        ksp.getResidualNorm()
                    uncached_results[key]['Converged Reason'] = \
                        KSPReasons[ksp.getConvergedReason()]

                    # If using gmres, estimate extreme singular values
                    compute_sing_val_params = set([
                        'ksp_compute_singularvalues',
                        'ksp_compute_eigenvalues',
                        'ksp_monitor_singular_value'])
                    if solver_params['ksp_type'] == 'gmres' and \
                            compute_sing_val_params & solver_params.keys():
                        emax, emin = ksp.computeExtremeSingularValues()
                        uncached_results[key]['Min Extreme Singular Value'] = \
                            emin
                        uncached_results[key]['Max Extreme Singular Value'] = \
                            emax

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
                    if mesh_dim == 2:
                        c = 0.5  # pylint: disable=C0103
                    else:
                        c = 0.75  # pylint: disable=C0103
                    print('Epsilon= %.2f^(%d+1) = %e'
                          % (c, fmm_order, c**(fmm_order+1)))

                print("L2 Err: ", l2_err)
                print("H1 Err: ", h1_err)
                print()

                # write to cache if necessary (after every computation)
                if uncached_results:
                    logger.info("Writing to cache...")

                    write_header = False  # pylint: disable=C0103
                    if write_over_duplicate_trials:
                        out_file = open(cache_file_name, 'w')
                        write_header = True
                    else:
                        if not os.path.isfile(cache_file_name):
                            write_header = True
                        out_file = open(cache_file_name, 'a')

                    cache_writer = csv.DictWriter(out_file, field_names)

                    if write_header:
                        cache_writer.writeheader()

                    # {{{ Move data to cache dictionary and append to file
                    #     if not writing over duplicates
                    for key in uncached_results:
                        if key in cache and not write_over_duplicate_trials:
                            out_file.close()
                            raise ValueError('Duplicating trial, maybe set'
                                             ' write_over_duplicate_trials to *True*?')

                        row = dict(key)
                        for output in uncached_results[key]:
                            row[output] = uncached_results[key][output]

                        if not write_over_duplicate_trials:
                            cache_writer.writerow(row)
                        cache[key] = uncached_results[key]

                    uncached_results = {}

                    # }}}

                    # {{{ Re-write all data if writing over duplicates

                    if write_over_duplicate_trials:
                        for key in cache:
                            row = dict(key)
                            for output in cache[key]:
                                row[output] = cache[key][output]
                            cache_writer.writerow(row)

                    # }}}

                    out_file.close()

                    logger.info("cache closed")
    del mesh
