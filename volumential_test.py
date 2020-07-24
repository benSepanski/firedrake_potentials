import pyopencl as cl
import logging

from firedrake import (
    UnitSquareMesh, SpatialCoordinate, FunctionSpace, Function,
    VolumePotential)


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
    m = UnitSquareMesh(10, 10)
    xx = SpatialCoordinate(m)

    V = FunctionSpace(m, 'CG', 1)
    f = Function(V).interpolate(xx[0] * xx[1])

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(m.geometric_dimension())

    # Build VolumePotential external operator
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    potential_data = {
        'kernel': kernel,
        'kernel_type': "Laplace",
        'cl_ctx': cl_ctx,
        'queue': queue,
        'nlevels': 5,
        'm_order': 10,
        'dataset_filename': "laplace.hdf5",
        'root_extent': 2,
        'table_kwargs': {
            'force_recompute': True,
            'n_brick_quad_points': 120,
            'adaptive_level': False,
            'use_symmetry': False,
            'alpha': 0,
            'n_levels': 1,
            },
        }

    logger.info("Creating volume potential")
    f_pot = VolumePotential(f, V, operator_data=potential_data)


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
