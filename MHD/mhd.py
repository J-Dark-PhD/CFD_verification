from fenics import *

N = 15
mesh = UnitCubeMesh(N, N, N)

V_ele = VectorElement("CG", mesh.ufl_cell(), 3)
Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele, Q_ele]))

fixed_wall_locations = "near(x[1], 0.) | near(x[1], 1.) | near(x[2], 0.) | near(x[2], 1.)"
inlet = "near(x[0], 0.)"
outlet = "near(x[0], 1.)"
hartmann_walls = "near(x[1], 0.) | near(x[1], 1.)"

velocity_inlet = DirichletBC(W.sub(0), Constant((10, 0, 0)), inlet)
pressure_outlet = DirichletBC(W.sub(1), Constant(0), outlet)
noslip = DirichletBC(W.sub(0), (0, 0, 0), fixed_wall_locations)
fully_conductive_walls = DirichletBC(W.sub(2), Constant(0), hartmann_walls)
bcu_insulated = [velocity_inlet, pressure_outlet, noslip]
bcu_conductive = [velocity_inlet, pressure_outlet, noslip, fully_conductive_walls]

func = Function(W)
export = Function(W)
u, p, phi = split(func)
v, q, q_2 = TestFunctions(W)
B = Constant((0, -1, 0))
Ha = Constant(30)
N = Ha**2

func_old = Function(W)
export = Function(W)

# Momentum
# F = (
# inner(dot(grad(u), u), v) * dx
# - inner(p, div(v)) * dx
# + inner(grad(u), grad(v)) * dx
# + N * (inner(cross(B, grad(phi)), v) * dx
# + inner(u * dot(B, B), v) * dx
# - inner(B * dot(B, u), v) * dx)
# )

# # CFD continuity
# F += - inner(q, div(u)) * dx

# # electric continuity
# F += (inner(grad(phi), grad(q_2)) * dx
# - inner(dot(B, curl(u)) + dot(u, curl(B)), q_2) * dx
# )

# solve(F == 0, func, bcu, solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})

# func_old.assign(func)
# export.assign(func)
# u_export, p_export, phi_export = export.split()
# XDMFFile("Results/u.xdmf".format(Ha)).write(u_export)
# XDMFFile("Results/phi.xdmf".format(Ha)).write(phi_export)
# XDMFFile("Results/p.xdmf".format(Ha)).write(p_export)

insulated_test_values = [0, 10, 30, 60, 100, 500, 1000]
conductive_test_values = [0, 10, 30, 60, 100]

for Hartmann_number in insulated_test_values:
    print("Running insulated case Ha={}".format(Hartmann_number))
    func.assign(func_old)    
    Ha = Constant(Hartmann_number)
    N = Ha**2

    # Momentum
    F = (
    inner(dot(grad(u), u), v) * dx
    - inner(p, div(v)) * dx
    + inner(grad(u), grad(v)) * dx
    + N * (inner(cross(B, grad(phi)), v) * dx
    + inner(u * dot(B, B), v) * dx
    - inner(B * dot(B, u), v) * dx)
    )

    # CFD continuity
    F += - inner(q, div(u)) * dx

    # electric continuity
    F += (inner(grad(phi), grad(q_2)) * dx
    - inner(dot(B, curl(u)) + dot(u, curl(B)), q_2) * dx
    )

    solve(F == 0, func, bcu_insulated, solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})

    func_old.assign(func)
    export.assign(func)
    u_export, p_export, phi_export = export.split()
    results_folder = "Results/fully_insulated/Ha={}/".format(Hartmann_number)
    XDMFFile(results_folder + "u.xdmf").write(u_export)
    XDMFFile(results_folder + "p.xdmf").write(p_export)
    XDMFFile(results_folder + "phi.xdmf").write(phi_export)


for Hartmann_number in conductive_test_values:
    print("Running conductive case Ha={}".format(Hartmann_number))
    func.assign(func_old)    
    Ha = Constant(Hartmann_number)
    N = Ha**2

    # Momentum
    F = (
    inner(dot(grad(u), u), v) * dx
    - inner(p, div(v)) * dx
    + inner(grad(u), grad(v)) * dx
    + N * (inner(cross(B, grad(phi)), v) * dx
    + inner(u * dot(B, B), v) * dx
    - inner(B * dot(B, u), v) * dx)
    )

    # CFD continuity
    F += - inner(q, div(u)) * dx

    # electric continuity
    F += (inner(grad(phi), grad(q_2)) * dx
    - inner(dot(B, curl(u)) + dot(u, curl(B)), q_2) * dx
    )

    solve(F == 0, func, bcu_conductive, solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})

    func_old.assign(func)
    export.assign(func)
    u_export, p_export, phi_export = export.split()
    results_folder = "Results/fully_conductive/Ha={}/".format(Hartmann_number)
    XDMFFile(results_folder + "u.xdmf").write(u_export)
    XDMFFile(results_folder + "p.xdmf").write(p_export)
    XDMFFile(results_folder + "phi.xdmf").write(phi_export)