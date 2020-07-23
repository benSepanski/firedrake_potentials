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

    # build sumpy kernel for sigma * e^{-r/alpha}
    logger.info("Setting up sumpy")
    sigma = 1.0
    alpha = 1.0
    dim = m.geometric_dimension()

    from math import e
    from pymbolic.primitives import make_sym_vector, Power
    from sumpy.symbolic import pymbolic_real_norm_2
    r = pymbolic_real_norm_2(make_sym_vector("d", dim))
    expr = Power(e, -r / alpha)

    from sumpy.kernel import ExpressionKernel
    kernel = ExpressionKernel(dim,
                              expr,
                              global_scaling_const=sigma,
                              is_complex_valued=False)

    # Build VolumePotential external operator
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    potential_data = {
        'kernel': kernel,
        'kernel_type': "ScaledGaussian",
        'cl_ctx': cl_ctx,
        'queue': queue,
        'nlevels': 5,
        'm_order': 10,
        'dataset_filename': "gaussianpotential.hdf5",
        'table_kwargs': {
            'n_brick_quad_points': 120,
            'adaptive_level': False,
            'use_symmetry': True,
            'alpha': 0,
            'n_levels': 1,
            },
        }

    logger.info("Creating volume potential")
    f_pot = VolumePotential(f, V, operator_data=potential_data)


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
