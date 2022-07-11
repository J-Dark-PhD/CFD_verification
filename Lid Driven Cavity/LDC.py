from fenics import *

# ##### Mesh parameters ##### #
N = 128
mesh = UnitSquareMesh(N, N, diagonal="crossed")

# ##### Define function spaces ##### #

V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele]))

# ##### Boundary conditions ##### #
lid_location = "near(x[1],  1.)"
fixed_wall_locations = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"

noslip = DirichletBC(W.sub(0), (0, 0), fixed_wall_locations)

u_x = 1
velocity_x = Constant((u_x, 0.0))
top_velocity = DirichletBC(W.sub(0), velocity_x,
                           lid_location)

bcu = [top_velocity, noslip]

# ##### Define variational parameters ##### #

v, q = TestFunctions(W)
up = Function(W)
u_export = Function(W)
up_old = Function(W)
u, p = split(up)

f = Constant((0, 0))


for top_velocity in [400, 1000, 2500, 5000, 10000]:

    up.assign(up_old)
    rho_0 = 1
    mu = 1
    # top_velocity = 1
    Reynolds = rho_0*top_velocity/mu
    print("Doing it for Re={}".format(Reynolds))
    velocity_x.assign(Constant((top_velocity, 0)))

    # Solver

    F = (
        rho_0*inner(grad(u)*u, v)*dx + mu*inner(grad(u), grad(v))*dx
        - inner(p, div(v))*dx + inner(q, div(u))*dx
        + inner(f, v)*dx
    )

    F += inner(q, div(u))*dx

    solve(F == 0, up, bcu)

    up_old.assign(up)
    u_export.assign(up)
    u_export, p_export = u_export.split()
    XDMFFile("Results/Re={:.1f}_u.xdmf".format(Reynolds)).write(u_export)
    XDMFFile("Results/Re={:.1f}_p.xdmf".format(Reynolds)).write(p_export)


# ##### Adaptive Mesh Refinement Option ##### #
# M = u[0]**2*dx
# epsilon_M = 1.e-6
# epsilon_M = Re*1.e-3

# solver = AdaptiveNonlinearVariationalSolver(problem, M)
# solver.parameters["nonlinear_variational_solver"]['newton_solver']["maximum_iterations"] = 10

# solver.solve(epsilon_M)