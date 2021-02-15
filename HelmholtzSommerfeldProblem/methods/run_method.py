import numpy as np

from math import log
from firedrake import FunctionSpace, VectorFunctionSpace, Function, grad, \
    TensorFunctionSpace, SpatialCoordinate

from .pml import pml
from .nonlocal_integral_eq import nonlocal_integral_eq
from .transmission import transmission


trial_options = set(['mesh', 'degree', 'kappa'])

method_required_options = {'pml': set(['pml_min',
                                       'pml_max',
                                       ]),
                           'nonlocal': set(['queue']),
                           'transmission': set([])}

# All have the options arguments 'options_prefix' and
# 'solver_parameters'
method_options = {'pml': ['pml_type',
                          'quad_const',
                          'speed'],
                  'nonlocal': ['FMM Order',
                               'FMM Tol',
                               'qbx_order',
                               'fine_order',
                               ],
                  'transmission': []}


prepared_trials = {}


def get_true_sol(fspace, kappa, cl_ctx, queue):
    """
    Get the ufl expression for the true solution (3D)
    or a function with the evaluated solution (2D)
    """
    mesh_dim = fspace.mesh().geometric_dimension()
    if mesh_dim == 3:
        spatial_coord = SpatialCoordinate(fspace.mesh())
        x, y, z = spatial_coord  # pylint: disable=C0103
        norm = sqrt(x**2 + y**2 + z**2)
        return Constant(1j / (4*pi)) / norm * exp(1j * kappa * norm)

    if mesh_dim == 2:
        # Evaluate true-sol using sumpy
        from sumpy.p2p import P2P
        from sumpy.kernel import HelmholtzKernel
        # https://github.com/inducer/sumpy/blob/900745184d2618bc27a64c847f247e01c2b90b02/examples/curve-pot.py#L87-L88
        p2p = P2P(cl_ctx, [HelmholtzKernel(dim=2)], exclude_self=False,
                  value_dtypes=np.complex128)
        # source is just (0, 0)
        sources = np.array([[0.0],[0.0]])
        strengths = np.array([[1.0], [1.0]])
        # targets are everywhere
        targets = np.array([Function(fspace).interpolate(x_i).dat.data
                            for x_i in SpatialCoordinate(fspace.mesh())])
        evt, (true_sol_arr,) = p2p(queue, targets, sources, strengths, k=kappa)
        true_sol = Function(fspace)
        true_sol.dat.data[:] = true_sol_arr[:]
        return true_sol
    raise ValueError("Only meshes of dimension 2, 3 supported")


def trial_to_tuple(trial):
    return (trial['mesh'], trial['degree'], trial['kappa'])


def prepare_trial(trial, true_sol_name, cl_ctx, queue):
    tuple_trial = trial_to_tuple(trial)
    if tuple_trial not in prepared_trials:

        mesh = trial['mesh']
        degree = trial['degree']
        kappa = trial['kappa']

        function_space = FunctionSpace(mesh, 'CG', degree)
        vect_function_space = VectorFunctionSpace(mesh, 'CG', degree)

        true_sol = get_true_sol(function_space, kappa, cl_ctx, queue)
        true_sol = Function(function_space).interpolate(true_sol)
        prepared_trials[tuple_trial] = (mesh, function_space, vect_function_space,
                                        true_sol, grad(true_sol))

    return prepared_trials[tuple_trial]


memoized_objects = {}


def run_method(trial, method,
               cl_ctx=None,
               queue=None,
               clear_memoized_objects=False,
               true_sol_name="True Solution",
               comp_sol_name="Computed Solution", **kwargs):
    """
        Returns (true solution, computed solution, snes_or_ksp)

        :arg clear_memoized_objects: Destroy memoized objects if true.
        :arg trial: A dict mapping each trial option to a valid value
        :arg method: A valid method (see the keys of *method_options*)
        :arg cl_ctx: the computing context
        :arg queue: the computing queue for the context

        kwargs should include the boundary id of the scatterer as 'scatterer_bdy_id'
        and the boundary id of the outer boundary as 'outer_bdy_id'

        kwargs should include the method options for :arg:`trial['method']`.
        for the given method.
    """
    if clear_memoized_objects:
        global memoized_objects
        memoized_objects = {}

    if cl_ctx is None:
        raise ValueError("Missing cl_ctx")
    if queue is None:
        raise ValueError("Missing queue")

    # Get boundary ids
    scatterer_bdy_id = kwargs['scatterer_bdy_id']
    outer_bdy_id = kwargs['outer_bdy_id']

    # Get degree and wave number
    degree = trial['degree']
    wave_number = trial['kappa']

    # Get options prefix and solver parameters, if any
    options_prefix = kwargs.get('options_prefix', None)
    solver_parameters = dict(kwargs.get('solver_parameters', None))

    # Get prepared trial args in kwargs
    prepared_trial = prepare_trial(trial, true_sol_name, cl_ctx, queue)
    mesh, fspace, vfspace, true_sol, true_sol_grad_expr = prepared_trial

    # Create a place to memoize any objects if necessary
    tuple_trial = trial_to_tuple(trial)
    memo_key = tuple_trial[:2]
    if memo_key not in memoized_objects:
        memoized_objects[memo_key] = {}

    comp_sol = None

    # Handle any special kwargs and get computed solution
    if method == 'pml':
        # Get required objects
        pml_max = kwargs['pml_max']
        pml_min = kwargs['pml_min']

        # Get optional argumetns
        pml_type = kwargs.get('pml_type', None)
        quad_const = kwargs.get('quad_const', None)
        speed = kwargs.get('speed', None)

        # Make tensor function space
        if 'tfspace' not in memoized_objects[memo_key]:
            memoized_objects[memo_key]['tfspace'] = \
                TensorFunctionSpace(mesh, 'CG', degree)

        tfspace = memoized_objects[memo_key]['tfspace']

        snes, comp_sol = pml(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                             options_prefix=options_prefix,
                             solver_parameters=solver_parameters,
                             fspace=fspace, tfspace=tfspace,
                             true_sol_grad_expr=true_sol_grad_expr,
                             pml_type=pml_type, quad_const=quad_const,
                             speed=speed,
                             pml_min=pml_min,
                             pml_max=pml_max,
                             )
        snes_or_ksp = snes

    elif method == 'nonlocal':
        # Build DG spaces if not already built
        if 'dgfspace' not in memoized_objects[memo_key]:
            memoized_objects[memo_key]['dgfspace'] = \
                FunctionSpace(mesh, 'DG', degree)
        if 'dgvfspace' not in memoized_objects[memo_key]:
            memoized_objects[memo_key]['dgvfspace'] = \
                VectorFunctionSpace(mesh, 'DG', degree)

        dgfspace = memoized_objects[memo_key]['dgfspace']
        dgvfspace = memoized_objects[memo_key]['dgvfspace']

        # Get opencl array context
        from meshmode.array_context import PyOpenCLArrayContext
        actx = PyOpenCLArrayContext(queue)

        # Build connection fd -> meshmode if not already built
        if 'meshmode_src_connection' not in memoized_objects[memo_key]:
            from meshmode.interop.firedrake import build_connection_from_firedrake
            memoized_objects[memo_key]['meshmode_src_connection'] = \
                build_connection_from_firedrake(
                    actx,
                    dgfspace,
                    grp_factory=None,
                    restrict_to_boundary=scatterer_bdy_id)

        meshmode_src_connection = memoized_objects[memo_key]['meshmode_src_connection']

        # Set defaults for qbx kwargs
        qbx_order = kwargs.get('qbx_order', degree+2)
        fine_order = kwargs.get('fine_order', 4 * degree)
        fmm_order = kwargs.get('FMM Order', None)
        fmm_tol = kwargs.get('FMM Tol', None)
        # make sure got either fmm_order xor fmm_tol
        if fmm_order is None and fmm_tol is None:
            raise ValueError("At least one of 'fmm_order', 'fmm_tol' must not "
                             "be *None*")
        if fmm_order is not None and fmm_tol is not None:
            raise ValueError("At most one of 'fmm_order', 'fmm_tol' must not "
                             "be *None*")
        # if got fmm_tol, make a level-to-order
        fmm_level_to_order = None
        if fmm_tol is not None:
            if not isinstance(fmm_tol, float):
                raise TypeError("fmm_tol of type '%s' is not of type float" % type(fmm_tol))
            if fmm_tol <= 0.0:
                raise ValueError("fmm_tol of '%s' is less than or equal to 0.0" % fmm_tol)
            from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder 
            fmm_level_to_order = SimpleExpansionOrderFinder(fmm_tol)
        # Otherwise, make sure we got a valid fmm_order
        else:
            if not isinstance(fmm_order, int):
                if fmm_order != False:
                    raise TypeError("fmm_order of type '%s' is not of type int" % type(fmm_order))
            if fmm_order != False and fmm_order < 1:
                raise ValueError("fmm_order of '%s' is less than 1" % fmm_order)

        qbx_kwargs = {'qbx_order': qbx_order,
                      'fine_order': fine_order,
                      'fmm_order': fmm_order,
                      'fmm_level_to_order': fmm_level_to_order,
                      'fmm_backend': 'fmmlib',
                      }
        # }}}

        ksp, comp_sol = nonlocal_integral_eq(
            mesh, scatterer_bdy_id, outer_bdy_id,
            wave_number,
            options_prefix=options_prefix,
            solver_parameters=solver_parameters,
            fspace=fspace, vfspace=vfspace,
            true_sol_grad_expr=true_sol_grad_expr,
            actx=actx,
            dgfspace=dgfspace,
            dgvfspace=dgvfspace,
            meshmode_src_connection=meshmode_src_connection,
            qbx_kwargs=qbx_kwargs,
            )

        snes_or_ksp = ksp

    elif method == 'transmission':

        snes, comp_sol = transmission(mesh, scatterer_bdy_id, outer_bdy_id,
                                      wave_number,
                                      options_prefix=options_prefix,
                                      solver_parameters=solver_parameters,
                                      fspace=fspace,
                                      true_sol_grad_expr=true_sol_grad_expr,
                                      )
        snes_or_ksp = snes
    else:
        raise ValueError("Invalid method")

    comp_sol.rename(name=comp_sol_name)
    return true_sol, comp_sol, snes_or_ksp
