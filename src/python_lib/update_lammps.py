# in this script I will define the function to get MOVE structure, and remember the structure of MOVE is like [i, dx, dy, dz], i here is the particle id in our program but not the one in lammps!

from cell_list import Cell_update, Cell_create
import numpy as np
from init_lammps import init_POSCAR_lammps
from Force_short_range import Force_short
from Force_long_range import Force_long
import Force_short_range_njit
import Force_long_range_njit
import Force_long_range_njit_exact
import input_read_lammps

def update_POS_lammps(lmp, POSCAR, INCAR, CELL_LIST):
    NUM_particle = int(lmp.get_natoms())
    x = lmp.extract_atom("x")
    MOVE = np.zeros([NUM_particle, 4])
    id = lmp.extract_atom("id")
    for i in range(NUM_particle):
        id_i = int(id[i]) - 1
        q_i, x_i, y_i, z_i = POSCAR[id_i]
        x_i_new, y_i_new, z_i_new = [x[i][0], x[i][1], x[i][2]]
        dx_i = x_i_new - x_i
        dy_i = y_i_new - y_i
        dz_i = z_i_new - z_i
        MOVE[i] = [int(id_i), dx_i, dy_i, dz_i]
    POSCAR, CELL_LIST = Cell_update(POSCAR, INCAR, CELL_LIST, MOVE)
    return POSCAR, CELL_LIST

def rewrite_POS_lammps(lmp, PARCAR, INCAR):
    POSCAR = init_POSCAR_lammps(lmp, PARCAR)
    CELL_LIST = Cell_create(POSCAR, INCAR)
    return POSCAR, CELL_LIST

def update_Force(lmp, POSCAR, INCAR, CELL_LIST):
    F = lmp.extract_atom("f")
    id = lmp.extract_atom("id")
    Force_long_val = Force_long(POSCAR, INCAR, CELL_LIST)
    Force_short_val = Force_short(POSCAR, INCAR, CELL_LIST)
    NUM_particle = int(lmp.get_natoms())
    Force_dict = {}
    
    for i in range(NUM_particle):
        F_xl, F_yl, F_zl =  Force_long_val
        F_xs, F_ys, F_zs =  Force_short_val
        F_x = F_xl[i] + F_xs[i]
        F_y = F_yl[i] + F_ys[i]
        F_z = F_zl[i] + F_zs[i]
        Force_dict[str(i + 1)] = [F_x, F_y, F_z]
    for i in range(NUM_particle):
        id_i = str(id[i])
        for j in range(3):
            F[i][j] += Force_dict[id_i][j]
    return Force_dict

def update_Force_njit(lmp, POSCAR, INCAR, CELL_LIST):
    F = lmp.extract_atom("f")
    id = lmp.extract_atom("id")
    mode = INCAR["mode"]
    KCAR = input_read_lammps.LOAD_KCAR(INCAR)
    if mode == "exact":
        Force_long_val = Force_long_range_njit_exact.Force_long(POSCAR, INCAR, CELL_LIST, KCAR)
    if mode == "rbm":
        Force_long_val = Force_long_range_njit.Force_long(POSCAR, INCAR, CELL_LIST)
    Force_short_val = Force_short_range_njit.Force_short(POSCAR, INCAR, CELL_LIST)
    NUM_particle = int(lmp.get_natoms())
    Force_dict = {}
    for i in range(NUM_particle):
        F_xl, F_yl, F_zl =  Force_long_val
        F_xs, F_ys, F_zs =  Force_short_val
        F_x = F_xl[i] + F_xs[i]
        F_y = F_yl[i] + F_ys[i]
        F_z = F_zl[i] + F_zs[i]
        Force_dict[str(i + 1)] = [F_x, F_y, F_z]
    for i in range(NUM_particle):
        id_i = str(id[i])
        for j in range(3):
            F[i][j] += Force_dict[id_i][j]
    return Force_dict

def update_velocity_njit(lmp, POSCAR, PARCAR, INCAR, CELL_LIST):
    v = lmp.extract_atom("v")
    id = lmp.extract_atom("id")
    type = lmp.extract_atom("type")
    mode = INCAR["mode"]
    KCAR = input_read_lammps.LOAD_KCAR(INCAR)
    if mode == "exact":
        Force_long_val = Force_long_range_njit_exact.Force_long(POSCAR, INCAR, CELL_LIST, KCAR)
    if mode == "rbm":
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
        for j in range(3):
            v[i][j] += Force_dict[id_i][j] * INCAR['dt']/mass + q_i * E[j] * INCAR['dt'] / mass
    return Force_dict