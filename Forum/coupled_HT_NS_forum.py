from fenics import *

# ##### Mesh parameters ##### #

mesh = UnitSquareMesh(40, 40, diagonal="crossed")

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
u, p = split(up)

f = Constant((0, 0))

rho_0 = 1
mu = 1
u_x = 1
Reynolds = rho_0*u_x/mu

velocity_x.assign(Constant((u_x, 0)))

# Solver

F = (
     rho_0*inner(grad(u)*u, v)*dx + mu*inner(grad(u), grad(v))*dx
     - inner(p, div(v))*dx + inner(q, div(u))*dx
     + inner(f, v)*dx
)

F += inner(q, div(u))*dx

solve(F == 0, up, bcu)

u, p = up.split()
XDMFFile("Re={:.1f}_u.xdmf".format(Reynolds)).write(u)
XDMFFile("Re={:.1f}_p.xdmf".format(Reynolds)).write(p)
