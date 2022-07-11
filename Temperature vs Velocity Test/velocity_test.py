from fenics import *
import sympy as sp
import numpy as np


from context2 import FESTIM
import properties

# ##### Mesh Parameters ##### #

nx = 80
ny = 80
mesh = RectangleMesh(Point(0, 0), Point(0.1, 0.1), nx, ny, diagonal="crossed")
vm = MeshFunction("size_t", mesh, 2)
sm = MeshFunction("size_t", mesh, 1)

vm.set_all(1)
boundary_R = CompiledSubDomain('on_boundary && near(x[0], 0.1, tol)',
                               tol=1E-14)
boundary_L = CompiledSubDomain('on_boundary && near(x[0], 0, tol)',
                               tol=1E-14)
boundary_top = CompiledSubDomain('on_boundary && near(x[1], 0.1, tol)',
                               tol=1E-14)
boundary_bot = CompiledSubDomain('on_boundary && near(x[1], 0, tol)',
                               tol=1E-14)

id_left = 1
id_right = 2
id_top = 3
id_bot = 4

boundary_L.mark(sm, id_left)
boundary_R.mark(sm, id_right)
boundary_top.mark(sm, id_top)
boundary_bot.mark(sm, id_bot)

# ##### Define Function Spaces ##### #

V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
T_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele, T_ele]))
# W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele]))

# Z = FunctionSpace(mesh, T_ele)

# T = Function(Z)


# ##### CFD --> Boundary conditions ##### #

# User defined boundary conditions
inlet_t = 598.15  # units: K
inlet_v = Constant((1.27e-04, 0))  # units: ms-1
outlet_p = 0  # units: Pa
# velocity_inlet = "near(x[0],  0.)"
# pressure_outlet = "near(x[0],  1.)"
# fixed_wall_locations = "near(x[1], 0.) | near(x[1], 1.)"

inflow = DirichletBC(W.sub(0), inlet_v, sm, id_left)
outflow = DirichletBC(W.sub(1), Constant((outlet_p)), sm, id_right)
noslip_1 = DirichletBC(W.sub(0), Constant((0, 0)), sm, id_top)
noslip_2 = DirichletBC(W.sub(0), Constant((0, 0)), sm, id_bot)
temperature_top = DirichletBC(W.sub(2), Constant(620), sm, id_top)
temperature_bot = DirichletBC(W.sub(2), Constant(620), sm, id_bot)
temperature_inlet = DirichletBC(W.sub(2), Constant(inlet_t), sm, id_left)

bcu = [inflow, noslip_1, noslip_2, temperature_top, temperature_bot, temperature_inlet]
# ##### CFD --> Fluid Materials properties ##### #

# LiPb
cp_lipb = properties.Cp_lipb
rho_lipb = properties.rho_lipb
rho_0_lipb = properties.rho_0_lipb
thermal_cond_lipb = properties.thermal_cond_lipb
visc_lipb = properties.visc_lipb
beta_lipb = properties.beta_lipb


# ##### CFD --> Define Variational Parameters ##### #

# T = Function(Z)
# w = TestFunction(Z)

v, q, w = TestFunctions(W)
# v, q = TestFunctions(W)
upT = Function(W)
u_export = Function(W)
upT_old = Function(W)
u, p, T = split(upT)
# u, p = split(up)


# Fluid properties
T_0 = inlet_t

rho = rho_lipb(T)
# rho = 9686
# rho_0 = rho_0_lipb
rho_0 = 9808.2464435
# rho_0 = 9686
mu = visc_lipb(T)  # 2.2e-3
# mu = 2.2e-3
beta = beta_lipb(T)  # 0.00012
# beta = 0.00012
cp = cp_lipb(T)  # 188
# cp = 188
g_direction = Constant((0.00, -9.81))
thermal_cond = thermal_cond_lipb(T)
# thermal_cond = 20

# ##### Solver ##### #
dx = Measure("dx", subdomain_data=vm)
ds = Measure("ds", subdomain_data=sm)
k_ramp = Constant(0)
F = (
        # CFD part
        rho_0*inner(dot(grad(u), u), v)*dx
        + inner(p, div(v))*dx
        + mu*inner(grad(u), grad(v))*dx
        - k_ramp*rho_0*inner(g_direction, v)*dx
        + k_ramp*(beta*rho_0)*inner((T-T_0)*g_direction, v)*dx
        + inner(div(u), q)*dx
        # Heat Transfer part
        + rho*cp*inner(dot(grad(T), u), w)*dx
        + thermal_cond*inner(grad(T), grad(w))*dx
        # - 1e5*w*dx
    )
# F_thermal = (
#     - rho*cp*inner(dot(grad(T), u), w)*dx
#     + thermal_cond*inner(grad(T), grad(w))*dx
# )

for k_ramp_value in [0, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03,
                     1e-02, 1e-01, 1]:
    # for velocity_value in np.logspace(-5, 0, num=20):
    # for velocity_value in [1e-03, 2e-03, 4e-03, 6e-03, 8e-03]:
    # for velocity_value in [1e-03, 2e-03, 3e-3, 4e-03, 6e-03]:
    # for k_ramp_value in np.logspace(-10, 0, num=30):

    # print("Doing it for v = {:.1e}".format(velocity_value))
    print("Doing it for k_ramp = {:.1e}".format(k_ramp_value))

    upT.assign(upT_old)

    # inlet_v.assign(Constant((velocity_value, 0)))
    k_ramp.assign(k_ramp_value)

    # ##### Adaptive Mesh Refinement Option ##### #

    # JF = derivative(F, up, TrialFunction(W))
    # problem = NonlinearVariationalProblem(F, up, bcu, JF)

    # M = u[0]**2*dx
    # epsilon_M = 1.e-6
    # solver = AdaptiveNonlinearVariationalSolver(problem, M)
    # solver.parameters["nonlinear_variational_solver"]['newton_solver']["maximum_iterations"] = 10

    # solver.solve(epsilon_M)
    JF = derivative(F, upT, TrialFunction(W))
    problem = NonlinearVariationalProblem(F, upT, bcu, JF)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-9
    solver.solve()
    # solve(F == 0, upT, bcu)
    
    # solve(F_thermal == 0, T, bcT)

    # ##### Post Processing ##### #

    upT_old.assign(upT)
    u_export.assign(upT)
    u_export, p_export, T_export = u_export.split()
    # XDMFFile("Results/v={:.1e}_u.xdmf".format(velocity_value)).write(u_export)
    # XDMFFile("Results/v={:.1e}_p.xdmf".format(velocity_value)).write(p_export)
    # XDMFFile("Results/v={:.1e}_T.xdmf".format(velocity_value)).write(T)

    XDMFFile("Results/v={:.1e}_u.xdmf".format(k_ramp_value)).write(u_export)
    XDMFFile("Results/v={:.1e}_p.xdmf".format(k_ramp_value)).write(p_export)
    XDMFFile("Results/v={:.1e}_T.xdmf".format(k_ramp_value)).write(T_export)
# u, p, T = upT.leaf_node().split()
# # u, p, T = split(up.leaf_node())
# XDMFFile("u.xdmf").write(u)
# XDMFFile("p.xdmf").write(p)
# XDMFFile("T.xdmf").write(T)
