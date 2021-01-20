import numpy as np

from firedrake import Function, FacetNormal, TestFunction, assemble, inner, ds, \
    TrialFunction, grad, dx, Constant, SpatialCoordinate, VectorFunctionSpace, \
    project
from firedrake.petsc import PETSc, OptionsManager

from pytential.target import PointsTarget

from sumpy.kernel import HelmholtzKernel

from .preconditioners.two_D_helmholtz import AMGTransmissionPreconditioner


def get_target_points_and_indices(fspace, boundary_ids):
    """
    Get  the points from the function space which lie on the given boundary
    id as a pytential PointsTarget, and their indices into the
    firedrake function

    :return: (target_indices, target_points)
    """
    # if just passed an int, convert to an iterable of ints
    # so that just one case to deal with
    if isinstance(boundary_ids, int):
        boundary_ids = [boundary_ids]
    target_markers = set(boundary_ids)

    # Check that bdy ids are valid
    if not target_markers <= set(fspace.mesh().exterior_facets.unique_markers):
        raise ValueError(
            "The following bdy ids are not exterior facet ids: %s" %
            (target_markers - set(fspace.mesh().exterior_facets.unique_markers)))

    if not target_markers & set(fspace.mesh().exterior_facets.unique_markers):
        raise ValueError("No bdy ids are exterior facet ids")

    target_indices = set()
    for marker in target_markers:
        target_indices |= set(
            fspace.boundary_nodes(marker, 'topological'))
    target_indices = np.array(list(target_indices), dtype=np.int32)

    target_indices = np.array(target_indices, dtype=np.int32)
    # Get coordinates of nodes
    coords = SpatialCoordinate(fspace.mesh())
    function_space_dim = VectorFunctionSpace(
        fspace.mesh(),
        fspace.ufl_element().family(),
        degree=fspace.ufl_element().degree())

    coords = Function(function_space_dim).interpolate(coords)
    coords = np.real(coords.dat.data)

    target_pts = coords[target_indices]
    # change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
    target_pts = np.transpose(target_pts).copy()
    return (target_indices, PointsTarget(target_pts))


def nonlocal_integral_eq(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                         options_prefix=None, solver_parameters=None,
                         fspace=None, vfspace=None,
                         true_sol_grad=None,
                         actx=None,
                         dgfspace=None,
                         dgvfspace=None,
                         qbx_kwargs=None,
                         ):
    r"""
        see run_method for descriptions of unlisted args

        args:

        gamma and beta are used to precondition
        with the following equation:

        \Delta u - \kappa^2 \gamma u = 0
        (\partial_n - i\kappa\beta) u |_\Sigma = 0
    """
    # away from the excluded region, but firedrake and meshmode point
    # into
    pyt_inner_normal_sign = -1

    ambient_dim = mesh.geometric_dimension()

    # {{{ Build src and tgt 

    # Build connection fd -> meshmode near src boundary
    from meshmode.interop.firedrake import build_connection_from_firedrake
    meshmode_src_connection = build_connection_from_firedrake(
        actx,
        dgfspace,
        grp_factory=None,
        restrict_to_boundary=scatterer_bdy_id)
    # build connection meshmode near src boundary -> src boundary inside meshmode
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    from meshmode.discretization.connection import make_face_restriction
    factory = InterpolatoryQuadratureSimplexGroupFactory(dgfspace.finat_element.degree)
    src_bdy_connection = make_face_restriction(actx, meshmode_src_connection.discr, factory, scatterer_bdy_id)
    # source is a qbx layer potential
    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(src_bdy_connection.to_discr, **qbx_kwargs)

    # get target indices and point-set
    target_indices, target = get_target_points_and_indices(fspace, outer_bdy_id)

    # }}}

    # build the operations
    from pytential import bind, sym
    r"""
    ..math:

    x \in \Sigma

    grad_op(x) =
        \nabla(
            \int_\Gamma(
                u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y)
        )
    """
    grad_op = pyt_inner_normal_sign * sym.grad(
        ambient_dim, sym.D(HelmholtzKernel(ambient_dim),
                           sym.var("u"), k=sym.var("k"),
                           qbx_forced_limit=None))

    r"""
    ..math:

    x \in \Sigma

    op(x) =
        i \kappa \cdot
        \int_\Gamma(
            u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
    """
    op = pyt_inner_normal_sign * 1j * sym.var("k") * (
        sym.D(HelmholtzKernel(ambient_dim),
              sym.var("u"), k=sym.var("k"),
              qbx_forced_limit=None)
        )

    # bind the operations
    pyt_grad_op = bind((qbx, target), grad_op)
    pyt_op = bind((qbx, target), op)

    # }}}

    class MatrixFreeB(object):
        def __init__(self, A, pyt_grad_op, pyt_op, actx, kappa):
            """
            :arg kappa: The wave number
            """

            self.actx = actx 
            self.k = kappa
            self.pyt_op = pyt_op
            self.pyt_grad_op = pyt_grad_op
            self.A = A
            self.meshmode_src_connection = meshmode_src_connection

            # {{{ Create some functions needed for multing
            self.x_fntn = Function(fspace)

            # CG
            self.potential_int = Function(fspace)
            self.potential_int.dat.data[:] = 0.0
            self.grad_potential_int = Function(vfspace)
            self.grad_potential_int.dat.data[:] = 0.0
            self.pyt_result = Function(fspace)

            self.n = FacetNormal(mesh)
            self.v = TestFunction(fspace)

            # some meshmode ones
            self.x_mm_fntn = self.meshmode_src_connection.discr.empty(self.actx,
                                                                      dtype='c')

            # }}}

        def mult(self, mat, x, y):
            # Copy function data into the fivredrake function
            self.x_fntn.dat.data[:] = x[:]
            # Transfer the function to meshmode
            self.meshmode_src_connection.from_firedrake(project(self.x_fntn, dgfspace),
                                                        out=self.x_mm_fntn)
            # Restrict to boundary
            x_mm_fntn_on_bdy = src_bdy_connection(self.x_mm_fntn)

            # Apply the operation
            potential_int_mm = self.pyt_op(self.actx,
                                           u=x_mm_fntn_on_bdy,
                                           k=self.k)
            grad_potential_int_mm = self.pyt_grad_op(self.actx,
                                                     u=x_mm_fntn_on_bdy,
                                                     k=self.k)
            # Store in firedrake
            self.potential_int.dat.data[target_indices] = potential_int_mm.get()
            for dim in range(grad_potential_int_mm.shape[0]):
                self.grad_potential_int.dat.data[target_indices, dim] = grad_potential_int_mm[dim].get()

            # Integrate the potential
            r"""
            Compute the inner products using firedrake. Note this
            will be subtracted later, hence appears off by a sign.

            .. math::

                \langle
                    n(x) \cdot \nabla(
                        \int_\Gamma(
                            u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
                        )d\gamma(y)
                    ), v
                \rangle_\Sigma
                - \langle
                    i \kappa \cdot
                    \int_\Gamma(
                        u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
                    )d\gamma(y), v
                \rangle_\Sigma
            """
            self.pyt_result = assemble(
                inner(inner(self.grad_potential_int, self.n),
                      self.v) * ds(outer_bdy_id)
                - inner(self.potential_int, self.v) * ds(outer_bdy_id)
            )

            # y <- Ax - evaluated potential
            self.A.mult(x, y)
            with self.pyt_result.dat.vec_ro as ep:
                y.axpy(-1, ep)

    # {{{ Compute normal helmholtz operator
    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    r"""
    .. math::

        \langle
            \nabla u, \nabla v
        \rangle
        - \kappa^2 \cdot \langle
            u, v
        \rangle
        - i \kappa \langle
            u, v
        \rangle_\Sigma
    """
    a = inner(grad(u), grad(v)) * dx \
        - Constant(wave_number**2) * inner(u, v) * dx \
        - Constant(1j * wave_number) * inner(u, v) * ds(outer_bdy_id)

    # get the concrete matrix from a general bilinear form
    A = assemble(a).M.handle
    # }}}

    # {{{ Setup Python matrix
    B = PETSc.Mat().create()

    # build matrix context
    Bctx = MatrixFreeB(A, pyt_grad_op, pyt_op, actx, wave_number)

    # set up B as same size as A
    B.setSizes(*A.getSizes())

    B.setType(B.Type.PYTHON)
    B.setPythonContext(Bctx)
    B.setUp()
    # }}}

    # {{{ Create rhs

    # Remember f is \partial_n(true_sol)|_\Gamma
    # so we just need to compute \int_\Gamma\partial_n(true_sol) H(x-y)

    sigma = sym.make_sym_vector("sigma", ambient_dim)
    r"""
    ..math:

    x \in \Sigma

    grad_op(x) =
        \nabla(
            \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y)
        )
    """
    grad_op = pyt_inner_normal_sign * \
        sym.grad(ambient_dim, sym.S(HelmholtzKernel(ambient_dim),
                                    sym.n_dot(sigma),
                                    k=sym.var("k"), qbx_forced_limit=None))
    r"""
    ..math:

    x \in \Sigma

    op(x) =
        i \kappa \cdot
        \int_\Gamma(
            f(y) H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
        )
    """
    op = 1j * sym.var("k") * pyt_inner_normal_sign * \
        sym.S(HelmholtzKernel(ambient_dim),
              sym.n_dot(sigma),
              k=sym.var("k"),
              qbx_forced_limit=None)

    rhs_grad_op = bind((qbx, target), grad_op)
    rhs_op = bind((qbx, target), op)

    # Transfer to meshmode
    true_sol_grad_mm = meshmode_src_connection.from_firedrake(project(true_sol_grad, dgvfspace), actx=actx)
    true_sol_grad_mm = src_bdy_connection(true_sol_grad_mm)
    # Apply the operations
    f_grad_convoluted_mm = rhs_grad_op(actx, sigma=true_sol_grad_mm, k=wave_number)
    f_convoluted_mm = rhs_op(actx, sigma=true_sol_grad_mm, k=wave_number)
    # Transfer function back to firedrake
    f_grad_convoluted = Function(vfspace)
    f_convoluted = Function(fspace)
    f_grad_convoluted.dat.data[:] = 0.0
    f_convoluted.dat.data[:] = 0.0

    for dim in range(f_grad_convoluted_mm.shape[0]):
        f_grad_convoluted.dat.data[target_indices, dim] = f_grad_convoluted_mm[dim].get()
    f_convoluted.dat.data[target_indices] = f_convoluted_mm.get()

    r"""
        \langle
            f, v
        \rangle_\Gamma
        + \langle
            i \kappa \cdot \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y), v
        \rangle_\Sigma
        - \langle
            n(x) \cdot \nabla(
                \int_\Gamma(
                    f(y) H_0^{(1)}(\kappa |x - y|)
                )d\gamma(y)
            ), v
        \rangle_\Sigma
    """
    rhs_form = inner(inner(true_sol_grad, FacetNormal(mesh)),
                     v) * ds(scatterer_bdy_id) \
        + inner(f_convoluted, v) * ds(outer_bdy_id) \
        - inner(inner(f_grad_convoluted, FacetNormal(mesh)),
                v) * ds(outer_bdy_id)

    rhs = assemble(rhs_form)

    # {{{ set up a solver:
    solution = Function(fspace, name="Computed Solution")

    #       {{{ Used for preconditioning
    if 'gamma' in solver_parameters or 'beta' in solver_parameters:
        gamma = complex(solver_parameters.pop('gamma', 1.0))

        import cmath
        beta = complex(solver_parameters.pop('beta', cmath.sqrt(gamma)))

        p = inner(grad(u), grad(v)) * dx \
            - Constant(wave_number**2 * gamma) * inner(u, v) * dx \
            - Constant(1j * wave_number * beta) * inner(u, v) * ds(outer_bdy_id)
        P = assemble(p).M.handle

    else:
        P = A
    #       }}}

    # Set up options to contain solver parameters:
    ksp = PETSc.KSP().create()
    if solver_parameters['pc_type'] == 'pyamg':
        del solver_parameters['pc_type']  # We are using the AMG preconditioner

        pyamg_tol = solver_parameters.get('pyamg_tol', None)
        if pyamg_tol is not None:
            pyamg_tol = float(pyamg_tol)
        pyamg_maxiter = solver_parameters.get('pyamg_maxiter', None)
        if pyamg_maxiter is not None:
            pyamg_maxiter = int(pyamg_maxiter)
        ksp.setOperators(B)
        ksp.setUp()
        pc = ksp.pc
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(AMGTransmissionPreconditioner(wave_number,
                                                          fspace,
                                                          A,
                                                          tol=pyamg_tol,
                                                          maxiter=pyamg_maxiter,
                                                          use_plane_waves=True))
    # Otherwise use regular preconditioner
    else:
        ksp.setOperators(B, P)

    options_manager = OptionsManager(solver_parameters, options_prefix)
    options_manager.set_from_options(ksp)

    import petsc4py.PETSc
    petsc4py.PETSc.Sys.popErrorHandler()
    with rhs.dat.vec_ro as b:
        with solution.dat.vec as x:
            ksp.solve(b, x)
    # }}}

    return ksp, solution
