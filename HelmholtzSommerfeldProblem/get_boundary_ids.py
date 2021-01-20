from firedrake import *
import matplotlib.pyplot as plt

m = Mesh("meshes/coarse-circle_in_square.msh")
m.init()

for marker in m.exterior_facets.unique_markers:
    V = FunctionSpace(m, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(Constant(0.0), v) * dx
    sol = Function(V)
    solve(a == L, sol, bcs=[DirichletBC(V, 1.0, marker)])

    trisurf(sol)
    plt.title(str(marker))
plt.show()
