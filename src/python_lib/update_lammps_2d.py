from cell_list import Cell_update, Cell_create
import numpy as np
from init_lammps import init_POSCAR_lammps
from Force_short_range import Force_short
from Force_long_range import Force_long
import Force_short_range_njit
import Force_long_range_njit
import Force_long_range_njit_exact
import input_read_lammps
import update_lammps

def z_confined_2d(lmp, POSCAR, PARCAR, INCAR, CELL_LIST):
    x = lmp.extract_atom("x")
    NUM_particle = int(lmp.get_natoms())
    L_z = INCAR['L_z']
    for i in range(NUM_particle):
        x[i][2] = L_z/2
    POSCAR, CELL_LIST = update_lammps.update_POS_lammps(lmp, POSCAR, INCAR, CELL_LIST)
    return POSCAR, CELL_LIST


def update_velocity_njit_2d(lmp, POSCAR, PARCAR, INCAR, CELL_LIST):
    v = lmp.extract_atom("v")
    id = lmp.extract_atom("id")
    type = lmp.extract_atom("type")
    Force_long_val = Force_long_range_njit.Force_long(POSCAR, INCAR, CELL_LIST)
    Force_short_val = Force_short_range_njit.Force_short(POSCAR, INCAR, CELL_LIST)
    NUM_particle = int(lmp.get_natoms())
    Force_dict = {}
    E = INCAR['E']
    for i in range(NUM_particle):
        F_xl, F_yl, F_zl =  Force_long_val
        F_xs, F_ys, F_zs =  Force_short_val
        F_x = F_xl[i] + F_xs[i]
        F_y = F_yl[i] + F_ys[i]
        F_z = F_zl[i] + F_zs[i]
        Force_dict[str(i + 1)] = [F_x, F_y, F_z]
    for i in range(NUM_particle):
        id_i = str(id[i])
        type_i = str(type[i])
        mass = PARCAR[type_i][0]
        q_i = PARCAR[type_i][1]
        for j in range(2):
            v[i][j] += Force_dict[id_i][j] * INCAR['dt']/mass + q_i * E[j] * INCAR['dt'] / mass
        v[i][2] = 0  
    return Force_dict