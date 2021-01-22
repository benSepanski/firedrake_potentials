from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

#mesh_name = "meshes/ball_in_cube-rad1.0-side6.0.step"
#mesh_name = "meshes/circle_in_square-rad1.0-side6.0.step"
#mesh_name = "meshes/betterplane.step"
mesh_name = "meshes/annulus-inner_rad1.0-outer_rad2.0.step"
m = OpenCascadeMeshHierarchy(mesh_name, 1.0, 1, cache=False).meshes[0]
m.init()
from matplotlib.collections import PolyCollection
triplot(m, interior_kw={'alpha': 0.2}, boundary_kw={'alpha': 0.2})

for marker in m.exterior_facets.unique_markers:
    V = FunctionSpace(m, "CG", 1)
    f = Function(V).interpolate(Constant(0.0))
    DirichletBC(V, 1.0, marker).apply(f)
    if m.geometric_dimension() == 2:
        trisurf(f)
    else:
        V3 = VectorFunctionSpace(m, 'CG', 1)
        newCoords = Function(V3).interpolate(m.coordinates)
        newCoords.dat.data[:] *= f.dat.data[:, np.newaxis]
        newM = Mesh(newCoords)
        triplot(newM, interior_kw={'alpha': 0.2}, boundary_kw={'alpha': 0.2})
    plt.title(str(marker))


plt.show()
