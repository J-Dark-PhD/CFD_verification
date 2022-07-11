import os, sys, inspect
import sympy as sp
from context2 import FESTIM
from context2 import properties
from fenics import *
# ##### Mesh Parameters ##### #

# mesh = UnitSquareMesh(40, 40)
# nx = ny = 40
# mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny, diagonal="crossed")

mesh = Mesh()
XDMFFile("mesh_domains.xdmf").read(mesh)
volume_markers, surface_markers = FESTIM.meshing.read_subdomains_from_xdmf(
                                 mesh, "mesh_domains.xdmf",
                                 "mesh_boundaries.xdmf")


# ##### Define Function Spaces ##### #

V_ele = VectorElement("CG", mesh.ufl_cell(), 2)
Q_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
T_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V_ele, Q_ele, T_ele]))

# IDs for volumes and surfaces (must be the same as in xdmf files)
id_lipb = 11
id_structure = 12
id_first_wall = 13
id_pipes = 14


id_inlet = 6
id_outlet = 7
id_cooling_lipb = 8
id_cooling_eurofer = 9
id_plasma_facing_wall = 10

# ##### CFD --> Boundary conditions ##### #

# User defined boundary conditions
inlet_temperature = 598.15  # units: K
inlet_velocity = 1.27e-04  # units: ms-1
inlet_pressure = 5e05  # units: Pa
outlet_pressure = 0  # units: Pa

# Simulation boundary conditions
non_slip = Constant((0.0, 0.0, 0.0))

inflow = DirichletBC(W.sub(0), Constant((-inlet_velocity, 0.0, 0.0)), surface_markers,
                     id_inlet)

inner_wall_1 = DirichletBC(W.sub(0), non_slip, surface_markers,
                           id_cooling_lipb)
inner_wall_3 = DirichletBC(W.sub(0), non_slip, surface_markers,
                           id_cooling_eurofer)
inner_wall_4 = DirichletBC(W.sub(0), non_slip, surface_markers,
                           id_plasma_facing_wall)


pressure_outlet = DirichletBC(W.sub(1), Constant(0), surface_markers,
                              id_outlet)

bcu = [inflow, pressure_outlet, inner_wall_1, inner_wall_3,
       inner_wall_4]

g = Constant((0.0, -9.81, 0.0))
T_0 = inlet_temperature

# ##### CFD --> Define Variational Parameters ##### #

v, q, w = TestFunctions(W)
up = Function(W)
u, p, T = split(up)

# ##### CFD --> Fluid Materials properties ##### #

# LiPb
Cp_lipb = properties.Cp_lipb(T)
rho_lipb = properties.rho_lipb(T)
rho_0_lipb = properties.rho_0_lipb
thermal_cond_lipb = properties.thermal_cond_lipb(T)
visc_lipb = properties.visc_lipb(T)
beta_lipb = properties.beta_lipb(T)

# ##### Heat Transfer --> Parameters ##### #

folder = "Solution"

parameters = {
    "mesh_parameters": {
        "mesh_file": "mesh_domains.xdmf",
        "cells_file": "mesh_domains.xdmf",
        "facets_file": "mesh_boundaries.xdmf",
    },
    "materials": [
        {
            # Tungsten
            "D_0": properties.D_0_W,
            "E_D": properties.E_D_W,
            "S_0": properties.S_0_W,
            "E_S": properties.E_S_W,
            "thermal_cond": properties.thermal_cond_W,
            "heat_capacity": properties.Cp_W,
            "rho": properties.rho_W,
            "id": id_first_wall,
        },
        {
            # EUROfer
            "D_0": properties.D_0_eurofer,
            "E_D": properties.E_D_eurofer,
            "S_0": properties.S_0_eurofer,
            "E_S": properties.E_S_eurofer,
            "thermal_cond": 22,  #properties.thermal_cond_eurofer,
            "heat_capacity": properties.Cp_eurofer,
            "rho": properties.rho_eurofer,
            "id": id_structure,
        },
        {
            # Pipes
            "D_0": properties.D_0_eurofer,
            "E_D": properties.E_D_eurofer,
            "S_0": properties.S_0_eurofer,
            "E_S": properties.E_S_eurofer,
            "thermal_cond": 22,  #properties.thermal_cond_eurofer,
            "heat_capacity": properties.Cp_eurofer,
            "rho": properties.rho_eurofer,
            "id": id_pipes,
        },
        {
            # LiPb
            "D_0": properties.D_0_lipb,
            "E_D": properties.E_D_lipb,
            "S_0": properties.E_S_lipb,
            "E_S": properties.E_S_lipb,
            "thermal_cond": properties.thermal_cond_lipb,
            "heat_capacity": properties.Cp_lipb,
            "rho": properties.rho_lipb,
            "id": id_lipb,
        },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surfaces": [id_plasma_facing_wall, id_cooling_eurofer, id_cooling_lipb, id_inlet],
            "value": 0,#*(FESTIM.t>=1e5) + (c_surf*(FESTIM.t/1e5))*(FESTIM.t<1e5)
        },
        # {
        #     "type": "recomb",
        #     "surfaces": id_coolant_surf,
        #     "Kr_0": 2.9e-14,
        #     "E_Kr": 1.92,
        #     "order": 2,
        # },
        ],
    "temperature": {
        "type": "solve_stationary",
        "boundary_conditions": [
            {
                "type": "flux",
                "value": 0.5e6,
                "surfaces": id_plasma_facing_wall
            },
            {
                "type": "dc",
                "value": 325+273.15,
                "surfaces": id_inlet
            },
            # {
            #     "type": "dc",
            #     "value": ,
            #     "surfaces": id_outlet
            # },
            {
                "type": "convective_flux",
                "h_coeff": 12.53e+03,
                "T_ext": 295+273.15,
                "surfaces": id_cooling_lipb
            },
            {
                "type": "convective_flux",
                "h_coeff": 12.53e+03,
                "T_ext": 285+273.15,
                "surfaces": id_cooling_eurofer
            }
            ],
        "source_term": [
            {
                "value": 23.2e+06*sp.exp(-71.74*FESTIM.x),
                "volume": id_first_wall,
            },
            {
                "value": 7.53e+06*sp.exp(-8.98*FESTIM.x),
                "volume": id_structure,
            },
            {
                "value": 7.53e+06*sp.exp(-8.98*FESTIM.x),
                "volume": id_pipes,
            },
            {
                # "value": 9.46e+06*sp.exp(-6.20*FESTIM.x),
                "value": 25.53e+06*sp.exp(-0.5089*FESTIM.x*100) + 5.443e+06*sp.exp(-0.0879*FESTIM.x*100),
                "volume": id_lipb,
            },
        ],
    },
}

bc_thermal, expressions = FESTIM.boundary_conditions.define_dirichlet_bcs_T(
                         parameters, W.sub(2), surface_markers)

bcu += bc_thermal

# Fluid properties
rho_0 = rho_0_lipb
mu = visc_lipb
beta = beta_lipb

# ##### Solver ##### #
dx = Measure("dx", subdomain_data=volume_markers)
ds = Measure("ds", subdomain_data=surface_markers)

F = (
    #           momentum
    rho_0*inner(grad(u), grad(v))*dx(id_lipb)
    + inner(p, div(v))*dx(id_lipb)
    + mu*inner(dot(grad(u), u), v)*dx(id_lipb)
    # - rho_0*inner(g, v)*dx(id_lipb)
    + (beta*rho_0)*inner((T-T_0)*g, v)*dx(id_lipb)
    #           continuity
    + inner(div(u), q)*dx(id_lipb)
)

# Velocity and Pressure is zero everywhere else
F += inner(u, v)*dx(id_structure)
F += inner(Constant((0, 0, 0)), v)*dx(id_structure)
F += inner(p, q)*dx(id_structure)
F += inner(Constant(0), q)*dx(id_structure)

F += inner(u, v)*dx(id_first_wall)
F += inner(Constant((0, 0, 0)), v)*dx(id_first_wall)
F += inner(p, q)*dx(id_first_wall)
F += inner(Constant(0), q)*dx(id_first_wall)

F += inner(u, v)*dx(id_pipes)
F += inner(Constant((0, 0, 0)), v)*dx(id_pipes)
F += inner(p, q)*dx(id_pipes)
F += inner(Constant(0), q)*dx(id_pipes)


# Solve for heat transfer
F_thermal, expressions = FESTIM.formulations.define_variational_problem_heat_transfers(
                        parameters, functions=(T, w), measurements=[dx, ds], dt=None)


F += F_thermal  # add thermal form to global form

# ##### Adaptive Mesh Refinement Option ##### #

# JF = derivative(F, up, TrialFunction(W))
# problem = NonlinearVariationalProblem(F, up, bcu, JF)

# M = u[0]**2*dx
# epsilon_M = 1.e-6
# solver = AdaptiveNonlinearVariationalSolver(problem, M)
# solver.parameters["nonlinear_variational_solver"]['newton_solver']["maximum_iterations"] = 10

# solver.solve(epsilon_M)
solve(F == 0, up, bcu)

# ##### Post Processing ##### #

u, p, T = up.leaf_node().split()
# u, p, T = split(up.leaf_node())
XDMFFile("u.xdmf").write(u)
XDMFFile("p.xdmf").write(p)
XDMFFile("T.xdmf").write(T)
