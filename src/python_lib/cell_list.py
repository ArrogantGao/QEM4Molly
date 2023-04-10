import numpy as np
from numpy import sqrt
from math import floor, ceil
from random import random

# In this script,functions used to create and update cell_list in XOY plane and Z axis will be defined
# XY_Cell_create(POSCAR, INCAR) will return 2d array
# Z_Cell_create(POSCAR, INCAR) will return 1d array
# Cell_create(POSCAR, INCAR) will return a dict CELL_LIST =  {'XY_Cell': XY_Cell, 'Z_Cell':Z_Cell}

# Cell_updata(POSCAR, INCAR, CELL_LIST, MOVE) will update the cell list, MOVE is an array:[..., [i, delta_x_i, delta_y_i, delta_z_i], ...], and return the moved POSCAR and CELL_LIST
# XY_Cell_update_once(POSCAR, INCAR, CELL_LIST, move) will move one particle in xOy plane and return the POSCAR and CELL_LIST
# Z_Cell_update_once(POSCAR, INCAR, CELL_LIST, move) will move one particle in z axis and return the POSCAR and CELL_LIST

#NOTICE!! Memeber in Z_Cell are listed from 0 to L_z when room number increase
# SO as to XY_Cell, and it should be written as XY_Cell[X][Y]

def sort_room_member(Z_cell_room, POSCAR):
    room_member_value = []
    for room_member in Z_cell_room:
        room_member_value.append([room_member, POSCAR[room_member][3]])
    sorted_room_member_value = sorted(room_member_value, key=lambda z: z[1], reverse = False)
    sorted_room_member = [value[0] for value in sorted_room_member_value]
    return sorted_room_member

def Cell_create(POSCAR, INCAR):

    NUM_particle = INCAR['NUM_particle']
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    L_z = INCAR['L_z']
    alpha = INCAR['alpha']
    r_c = INCAR['r_c']

    X_Cell_num, Y_Cell_num, Z_Cell_num, X_Cell_length, Y_Cell_length, Z_Cell_length = INCAR['Cell_info']

    XY_Cell = [[[] for i in range(Y_Cell_num)] for j in range(X_Cell_num)]
    Z_Cell = [[] for i in range(Z_Cell_num)]

    for i in range(NUM_particle):
        q_i, x_i, y_i, z_i = POSCAR[i]
        X_Cell_roomnum = floor(x_i/X_Cell_length) % X_Cell_num
        Y_Cell_roomnum = floor(y_i/Y_Cell_length) % Y_Cell_num
        Z_Cell_roomnum = floor(z_i/Z_Cell_length) % Z_Cell_num
        
        XY_Cell[X_Cell_roomnum][Y_Cell_roomnum].append(i)
        Z_Cell[Z_Cell_roomnum].append(i)
    
    for roomnum in range(Z_Cell_num):
        Z_Cell_room = Z_Cell[roomnum]
        Z_Cell_room_new = sort_room_member(Z_Cell_room, POSCAR)
        Z_Cell[roomnum] = Z_Cell_room_new


    CELL_LIST = {'XY_Cell': XY_Cell, 'Z_Cell':Z_Cell}
    return CELL_LIST


def Cell_update(POSCAR, INCAR, CELL_LIST, MOVE):

    NUM_particle = INCAR['NUM_particle']
    L_x = INCAR['L_x']
    L_y = INCAR['L_y']
    L_z = INCAR['L_z']
    alpha = INCAR['alpha']
    r_c = INCAR['r_c']

    X_Cell_num, Y_Cell_num, Z_Cell_num, X_Cell_length, Y_Cell_length, Z_Cell_length = INCAR['Cell_info']

    XY_Cell = CELL_LIST['XY_Cell']
    Z_Cell = CELL_LIST['Z_Cell']

    MOVE_NUM = np.shape(MOVE)[0]
    
    for move_step in range(MOVE_NUM):
        move = MOVE[move_step]

        moved_particle_num = int(move[0])
        q, x_old, y_old, z_old = POSCAR[moved_particle_num]
        x_new = (x_old + move[1]) % L_x
        y_new = (y_old + move[2]) % L_y
        z_new = z_old + move[3]

        #NOITCE: the handle below is just an expediency, I still don't know how to deal with the z boundary in MD/MC sumilation#
        z_new = z_new % L_z
        ###################
        # now we know by adding a fix wall in lammps we do not need to add that in z direction, lammps will do that for us

        POSCAR[moved_particle_num] = [q, x_new, y_new, z_new]
        
        X_old_roomnum = floor(x_old/X_Cell_length) % X_Cell_num
        Y_old_roomnum = floor(y_old/Y_Cell_length) % Y_Cell_num
        Z_old_roomnum = floor(z_old/Z_Cell_length) % Z_Cell_num
        X_new_roomnum = floor(x_new/X_Cell_length) % X_Cell_num
        Y_new_roomnum = floor(y_new/Y_Cell_length) % Y_Cell_num
        Z_new_roomnum = floor(z_new/Z_Cell_length) % Z_Cell_num

        XY_Cell[X_old_roomnum][Y_old_roomnum].remove(moved_particle_num)
        Z_Cell[Z_old_roomnum].remove(moved_particle_num)

        XY_Cell[X_new_roomnum][Y_new_roomnum].append(moved_particle_num)
        Z_Cell[Z_new_roomnum].append(moved_particle_num)

        Z_Cell[Z_new_roomnum] = sort_room_member(Z_Cell[Z_new_roomnum], POSCAR)

    CELL_LIST['XY_Cell'] = XY_Cell
    CELL_LIST['Z_Cell'] = Z_Cell

    return POSCAR, CELL_LIST

