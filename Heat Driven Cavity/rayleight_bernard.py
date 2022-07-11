from fenics import *
import matplotlib.pyplot as plt

N = 128

M = UnitSquareMesh(N, N)
L = 1  # characteristic length
V_ele = VectorElement("CG", M.ufl_cell(), 2)
Q_ele = FiniteElement("CG", M.ufl_cell(), 1)
T_ele = FiniteElement("CG", M.ufl_cell(), 1)
Z = FunctionSpace(M, MixedElement([V_ele, Q_ele, T_ele]))

upT = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

T_1, T_0 = 1, 0

g = -9.81  # gravity acceleration in m/s2
mu = 1  # dynamic viscosity in kg/m/s
rho = 1  # density in kg/m3
rho_0 = 1
cp = 1  # heat capacity in J/(kg.K)
thermal_cond = 1  # thermal conductivity in W/(m.K)
nu = mu/rho  # kinematic visocity in m2/s
alpha = thermal_cond/rho/cp  # thermal diffusivity in m2/s
beta = 1  # thermal expansion in K-1
Ra = Constant(beta/nu/alpha*g*L**3)
Pr = Constant(mu*cp/thermal_cond)

# Ra = Constant(1e03)
# Pr = Constant(1)

beta = Constant(1e05)

g_direction = Constant((0, -1))
T_1 = Constant(T_1)
T_0 = Constant(T_0)

# F = (
#     inner(grad(u), grad(v))*dx
#     + inner(dot(grad(u), u), v)*dx
#     - inner(p, div(v))*dx
#     + (Ra/Pr)*inner((T-T_0)*g_direction, v)*dx
#     + inner(div(u), q)*dx
#     + rho*cp*inner(dot(grad(T), u), S)*dx
#     + thermal_cond*inner(grad(T), grad(S))*dx

# )

F = (
    # CFD part
    rho_0*inner(grad(u), grad(v))*dx
    - mu*inner(dot(grad(u), u), v)*dx
    - inner(p, div(v))*dx
    + (beta*rho_0)*inner((T-T_0)*g_direction, v)*dx
    + inner(dot(grad(u), u), v)*dx
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
u, p, T = upT.split()
folder = 'Ra_1e05'
XDMFFile(folder + '/u.xdmf').write(u)
XDMFFile(folder + '/T.xdmf').write(T)
XDMFFile(folder + '/p.xdmf').write(p)
