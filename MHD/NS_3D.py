from fenics import *

N = 10
mesh = UnitCubeMesh(N, N, N)

V_ele = VectorElement("CG", mesh.ufl_cell(), 3)
Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele]))

fixed_wall_locations = "near(x[1], 0.) | near(x[1],  1.) | near(x[2],  0.) | near(x[2],  1.)"
inlet = "near(x[0], 0.)"
outlet = "near(x[0], 1.)"
velocity_inlet = DirichletBC(W.sub(0), Constant((10, 0, 0)), inlet)
pressure_outlet = DirichletBC(W.sub(1), Constant(0), outlet)
noslip = DirichletBC(W.sub(0), (0, 0, 0), fixed_wall_locations)
bcu = [velocity_inlet, pressure_outlet, noslip]

func = Function(W)
export = Function(W)
u, p = split(func)
v, q = TestFunctions(W)

export = Function(W)

# Momentum
F = (
inner(dot(grad(u), u), v) * dx
- inner(p, div(v)) * dx
+ inner(grad(u), grad(v)) * dx
)

# CFD continuity
F += - inner(q, div(u)) * dx

solve(F == 0, func, bcu, solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})

export.assign(func)
u_export, p_export = export.split()
XDMFFile("Results/u.xdmf").write(u_export)
XDMFFile("Results/p.xdmf").write(p_export)

