from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import random
import numpy as np

def init_lammps(INCAR, PARCAR):
    lmp = lammps()
    
    #init the system
    #lmp.command('echo screen')
    lmp.command('units lj')
    lmp.command('atom_style full')
    lmp.command('boundary p p f')
    
    #define the variables
    lmp.command('variable L_x equal ' + str(INCAR['L_x']))
    lmp.command('variable L_y equal ' + str(INCAR['L_y']))
    lmp.command('variable L_z equal ' + str(INCAR['L_z']))
    lmp.command('variable LJ_eps equal ' + str(INCAR['LJ_eps']))
    lmp.command('variable LJ_sig equal ' + str(INCAR['LJ_sig']))
    lmp.command('variable LJ_cut equal ' + str(INCAR['LJ_cut']))
    lmp.command('variable dt equal ' + str(INCAR['dt']))
    lmp.command('variable LJ_bin equal ' + str(0.5 * INCAR['LJ_cut']))
    lmp.command('variable mid_lo equal ' + str(INCAR['wall_cut']))
    lmp.command('variable mid_hi equal ' + str(INCAR['L_z'] - INCAR['wall_cut']))
    lmp.command('variable temp equal ' + str(INCAR['T']))
    
    #create the box
    lmp.command('region box block 0 ${L_x} 0 ${L_y} 0 ${L_z}')
    lmp.command('region middle block 0 ${L_x} 0 ${L_y} ${mid_lo} ${mid_hi}')
    
    #create atoms
    NUM_TYPE = len(PARCAR)
    lmp.command('create_box ' + str(NUM_TYPE) + ' box')
    for num_type in range(1, NUM_TYPE + 1):
        mass, charge, num_particle = PARCAR[str(num_type)]
        lmp.command('create_atoms ' + str(num_type) + ' random ' + str(num_particle) + ' ' + str(random.randint(1, 10000)) + ' middle')
        lmp.command('mass ' + str(num_type) + ' ' + str(mass))
                    
    lmp.file('./INPUT/lammps.in')

    return lmp

def init_POSCAR_lammps(lmp, PARCAR):
    x = lmp.extract_atom("x")
    NUM_particle = lmp.get_natoms()
    POSCAR = np.zeros([NUM_particle, 4])
    TYPE = lmp.extract_atom("type")
    id = lmp.extract_atom("id")
    for i in range(0, NUM_particle):
        type_i = TYPE[i]
        id_i = int(id[i]) - 1
        q_i = PARCAR[str(type_i)][1]
        x_i = x[i][0]
        y_i = x[i][1]
        z_i = x[i][2]
        POSCAR[id_i] = [q_i, x_i, y_i, z_i]
    return POSCAR