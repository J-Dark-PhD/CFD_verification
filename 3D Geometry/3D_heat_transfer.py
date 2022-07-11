import os, sys, inspect
import sympy as sp
from context2 import FESTIM
from context2 import properties
from FESTIM.generic_simulation import run

# IDs for volumes and surfaces (must be the same as in xdmf files)
id_lipb_flow_region = 11
id_structure = 12
id_first_wall = 13
id_pipes = 14


id_inlet = 6
id_outlet = 7
id_cooling_lipb = 8
id_cooling_eurofer = 9
id_plasma_facing_wall = 10

# ###### Parameters ##### #

folder = "3D_Solution"

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
            "thermal_cond": properties.thermal_cond_eurofer,
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
            "thermal_cond": properties.thermal_cond_eurofer,
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
            "id": id_lipb_flow_region,
        },
        ],
    "boundary_conditions": [
        {
            "type": "dc",
            "surfaces": [id_plasma_facing_wall, id_cooling_eurofer, id_cooling_lipb, id_inlet],
            "value": 0,#*(FESTIM.t>=1e5) + (c_surf*(FESTIM.t/1e5))*(FESTIM.t<1e5)
        },
        ],
    "traps": [],
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
                # "value": 7.53e+06*sp.exp(-8.98*FESTIM.x), # shimwell value
                "value": (9.6209e06*sp.exp(-12.02*FESTIM.x))*(FESTIM.x < 0.15) + (4.7109e06*sp.exp(-7.773*FESTIM.x))*(FESTIM.x >= 0.15),
                "volume": id_structure,
            },
            {
                # "value": 7.53e+06*sp.exp(-8.98*FESTIM.x), # shimwell value
                "value": (9.6209e06*sp.exp(-12.02*FESTIM.x))*(FESTIM.x < 0.15) + (4.7109e06*sp.exp(-7.773*FESTIM.x))*(FESTIM.x >= 0.15),
                "volume": id_pipes,
            },
            {
                # "value": 9.46e+06*sp.exp(-6.20*FESTIM.x)*(0.005*(FESTIM.x > 0.2) + 1 *(FESTIM.x <= 0.2)), # shimwell value
                "value": (3.9108e05*FESTIM.x**(-1.213))*(FESTIM.x < 0.15) + 8.4629e06*sp.exp(-5.485*FESTIM.x)*(FESTIM.x >= 0.15),
                "volume": id_lipb_flow_region,
            },
        ],
    },
     "solving_parameters": {
        "type": "solve_stationary",
        "traps_finite_element": 'DG',
        "final_time": 1e7,
        "initial_stepsize": 1e-5,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": 1e618,
            "stepsize_stop_max": 1/10,
            "dt_min": 1e4,
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 10,
        }
        },
    "exports": {
        "xdmf": {
            "functions": ['T', 'solute'],
            "labels": ['T', 'solute'],
            "folder": folder,
        },
    },
}


# ##### Solver ##### #

run(parameters, log_level=20)
        