from fenics import *
import matplotlib.pyplot as plt

N = 120

M = UnitSquareMesh(N, N)
V_ele = VectorElement("CG", M.ufl_cell(), 2)
Q_ele = FiniteElement("CG", M.ufl_cell(), 1)
T_ele = FiniteElement("CG", M.ufl_cell(), 1)
Z = FunctionSpace(M, MixedElement([V_ele, Q_ele, T_ele]))

upT = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

T_1, T_0 = 1, 0

# ##### Feeding function ##### #

upT = Function(Z)
u_export = Function(Z)
upT_old = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

# 1e03, 1e04, 1e05, 1e06, 5e06, 1e07, 5e07, 1e08

for beta in [1e05]:

    upT.assign(upT_old)

    # g = Constant((0, 9.81))  # gravity acceleration in m/s2
    mu = 1  # dynamic viscosity in kg/m/s
    rho = 1  # density in kg/m3
    rho_0 = 1
    cp = 1  # heat capacity in J/(kg.K)
    thermal_cond = 1  # thermal conductivity in W/(m.K)
    nu = mu/rho  # kinematic visocity in m2/s
    # beta = 1e05  # thermal expansion in K-1

    g = Constant((0, -1))
    Ra = beta
    print("Doing it for Ra={:.1e}".format(Ra))
    # print("Doing it for Rho_0 = {}".format(Ra))
    # print("g = {}".format(g))

    F = (
        # CFD part
        #           momentum
        rho_0*inner(grad(u), grad(v))*dx
        + inner(p, div(v))*dx
        + mu*inner(dot(grad(u), u), v)*dx
        - rho_0*inner(g, v)*dx
        + (beta*rho_0)*inner((T-T_0)*g, v)*dx
                #           continuity
        + inner(div(u), q)*dx
        # Heat Transfer part
        + rho*cp*inner(dot(grad(T), u), S)*dx
        + thermal_cond*inner(grad(T), grad(S))*dx
    )

    bcs = [
        DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary"),
        DirichletBC(Z.sub(2), T_1, "on_boundary && x[0] == 0"),
        DirichletBC(Z.sub(2), T_0, "on_boundary && x[0] == 1")
    ]   

    solve(F == 0, upT, bcs=bcs)

    upT_old.assign(upT)
    u_export.assign(upT)
    u_export, p_export, T_export = u_export.split()
    XDMFFile("Results/Ra={:.1e}_u.xdmf".format(Ra)).write(u_export)
    XDMFFile("Results/Ra={:.1e}_p.xdmf".format(Ra)).write(p_export)
    XDMFFile("Results/Ra={:.1e}_T.xdmf".format(Ra)).write(T_export)


    # u, p, T = upT.split()
    # folder = 'Ra_1e07'
    # XDMFFile(folder + '/u.xdmf').write(u)
    # XDMFFile(folder + '/T.xdmf').write(T)
    # XDMFFile(folder + '/p.xdmf').write(p)