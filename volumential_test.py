import logging
import numpy as np
import pyopencl as cl

from firedrake import (
    UnitSquareMesh, SpatialCoordinate, as_tensor,
    exp, FunctionSpace, Function,
    VolumePotential, sqrt, assemble, inner, dx)


# set up logging
verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.CRITICAL)


def main():
    # make function space and function
    logger.info("Building function space")
    m = UnitSquareMesh(32, 32)
    # get spatial coordinate, shifted so that [0,1]^2 -> [-0.5,0.5]^2
    xx = SpatialCoordinate(m)
    shifted_xx = as_tensor([xx[0] - 0.5, xx[1] - 0.5])

    alpha = 160
    norm2 = shifted_xx[0] * shifted_xx[0] + shifted_xx[1] * shifted_xx[1]
    source_expr = -(4 * alpha ** 2 * norm2 - 4 * alpha) * exp(-alpha * norm2)
    sol_expr = exp(-alpha * norm2)
    logger.info("source_expr : %s" % source_expr)
    logger.info("sol_expr : %s" % sol_expr)

    order = 8
    logger.info(f"Building FunctionSpace of order {order}")
    fspace = FunctionSpace(m, 'CG', order)
    logger.info("interpolating source and solution")
    source = Function(fspace).interpolate(source_expr)
    sol = Function(fspace).interpolate(sol_expr)

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(m.geometric_dimension())

    # We could set to a custom group factory if we wanted to,
    # defaults to recursive nodes with 'lgl' nodes
    #
    # from meshmode.discretization.poly_element import (
    #     PolynomialWarpAndBlendGroupFactory)
    # grp_factory = PolynomialWarpAndBlendGroupFactory(order)
    grp_factory = None

    # Build VolumePotential external operator
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    potential_data = {
        'kernel': kernel,
        'kernel_type': "Laplace",
        'cl_ctx': cl_ctx,
        'queue': queue,
        'nlevels': 6,
        'm_order': 20,
        'dataset_filename': f"laplace-order{order}.hdf5",
        'grp_factory': grp_factory,
        'root_extent': 2,
        'table_compute_method': "DrosteSum",
        'table_kwargs': {
            'force_recompute': False,
            'n_brick_quad_points': 100,
            'adaptive_level': False,
            'use_symmetry': True,
            'alpha': 0.1,
            'nlevels': 15,
            },
        'fmm_kwargs': {},
        }

    logger.info("Creating volume potential")
    pot = VolumePotential(source, fspace, operator_data=potential_data)
    logger.info("Evaluating potential")
    pot.evaluate(continuity_tolerance=1e-8)

    max_nodal_diff = np.max(np.abs(pot.dat.data - sol.dat.data))
    print("Max nodal difference: %e" % max_nodal_diff)


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
