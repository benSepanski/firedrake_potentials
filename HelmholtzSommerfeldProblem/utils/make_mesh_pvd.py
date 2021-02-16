from firedrake import *
m = Mesh('meshes/circle_in_square-rad1.0-side6.0-h5.00000e-01.msh')

from utils.to_2nd_order import to_2nd_order
m2 = to_2nd_order(m, 3, rad=1.0)

V = FunctionSpace(m2, 'DG', 2)

pmlSpongeSurface = 4
innerSurface = 5

oneInInner = Function(V).interpolate(Constant(1.0))

oneInInner.dat.data[V.cell_node_list[m2.cell_subset(pmlSpongeSurface).indices]] = 0.0

outFile = File("circle_in_square.pvd", 'w')
outFile.write(oneInInner)

"""
from meshmode.interop.firedrake import import_firedrake_mesh
m2mm, _ = import_firedrake_mesh(m2)

from meshmode.mesh.visualization import write_vertex_vtk_file
write_vertex_vtk_file(m2mm, 'circle_in_square.vtu')
"""
