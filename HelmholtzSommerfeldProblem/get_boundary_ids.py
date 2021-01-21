from firedrake import *
import matplotlib.pyplot as plt

m = Mesh("meshes/coarse-circle_in_square-rad1.0-side6.0.msh")
m.init()

for marker in m.exterior_facets.unique_markers:
    V = FunctionSpace(m, "CG", 1)
    f = Function(V).interpolate(Constant(0.0))
    DirichletBC(V, 1.0, marker).apply(f)
    trisurf(f)
    plt.title(str(marker))
plt.show()
