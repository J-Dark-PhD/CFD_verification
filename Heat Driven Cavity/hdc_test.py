from fenics import *
import matplotlib.pyplot as plt
from context2 import FESTIM
import properties

N = 120

M = UnitSquareMesh(N, N)
V_ele = VectorElement("CG", M.ufl_cell(), 2)
Q_ele = FiniteElement("CG", M.ufl_cell(), 1)
T_ele = FiniteElement("CG", M.ufl_cell(), 1)
Z = FunctionSpace(M, MixedElement([V_ele, Q_ele, T_ele]))

upT = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

T_1, T_0 = 700, 600

cp_lipb = properties.Cp_lipb
rho_lipb = properties.rho_lipb
rho_0_lipb = properties.rho_0_lipb
thermal_cond_lipb = properties.thermal_cond_lipb
visc_lipb = properties.visc_lipb
beta_lipb = properties.beta_lipb

# ##### Feeding function ##### #

upT = Function(Z)
u_export = Function(Z)
upT_old = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

# mu = 1  # dynamic viscosity in kg/m/s
# rho = 1  # density in kg/m3
# rho_0 = 1
# cp = 1  # heat capacity in J/(kg.K)
# thermal_cond = 1  # thermal conductivity in W/(m.K)
# nu = mu/rho  # kinematic visocity in m2/s
# # beta = 1e05  # thermal expansion in K-1

# mu = visc_lipb(T)
# rho = rho_lipb(T)
# rho_0 = rho_0_lipb
# cp = cp_lipb(T)
# thermal_cond = thermal_cond_lipb(T)
# beta = beta_lipb(T)

bcs = [
    DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary"),
    DirichletBC(Z.sub(2), T_1, "on_boundary && x[0] == 0"),
    DirichletBC(Z.sub(2), T_0, "on_boundary && x[0] == 1")
]   

for alpha in [1]:
# for gamma in [1, -1]:

    upT.assign(upT_old)

    print("Doing it for Alpha = {:.1e}".format(alpha))
    # print("Doing it for Gamma = {:.1e}".format(gamma))


    rho = 9686
    rho_0 = 9686
    mu = 2.2e-3
    beta = 0.00012
    cp = 188
    thermal_cond = 20
    gamma = 1

    g = Constant((0, -9.81))

    F = (
        # CFD part
        #           momentum
        rho_0*inner(grad(u), grad(v))*dx
        + inner(p, div(v))*dx
        + alpha*mu*inner(dot(grad(u), u), v)*dx
        - gamma*rho_0*inner(g, v)*dx
        + 50*(beta*rho_0)*inner((T-T_0)*g, v)*dx
        #           continuity
        + inner(div(u), q)*dx
        # Heat Transfer part
        + rho*cp*inner(dot(grad(T), u), S)*dx
        + thermal_cond*inner(grad(T), grad(S))*dx
    )

    solve(F == 0, upT, bcs=bcs)

    upT_old.assign(upT)
    u_export.assign(upT)
    u_export, p_export, T_export = u_export.split()
    XDMFFile("Results/alpha2={:.1e}_u.xdmf".format(alpha)).write(u_export)
    XDMFFile("Results/alpha2={:.1e}_p.xdmf".format(alpha)).write(p_export)
    XDMFFile("Results/alpha2={:.1e}_T.xdmf".format(alpha)).write(T_export)

    # XDMFFile("Results/gamma={:.1e}_u.xdmf".format(gamma)).write(u_export)
    # XDMFFile("Results/gamma={:.1e}_p.xdmf".format(gamma)).write(p_export)
    # XDMFFile("Results/gamma={:.1e}_T.xdmf".format(gamma)).write(T_export)

    # u, p, T = upT.split()
    # folder = 'legitness'
    # XDMFFile(folder + '/u.xdmf').write(u)
    # XDMFFile(folder + '/T.xdmf').write(T)
    # XDMFFile(folder + '/p.xdmf').write(p)
