from lammps import lammps
from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import random
import numpy as np

#In this script I will deifin the Andersen themrostat, which is used to correct the tempeture of the system before each 'run 1' command

def Andersen_themrostat(lmp, INCAR, PARCAR):
    nu = INCAR['nu']
    T = INCAR['T']
    dt = INCAR['dt']
    sigma = np.sqrt(T)
    v = lmp.extract_atom("v")
    N = lmp.get_natoms()
    type = lmp.extract_atom("type")
    for i in range(N):
        if (random.random() < nu * dt):
            type_i = str(type[i])
            mass = PARCAR[type_i][0]
            sigma = np.sqrt(T/mass)
            v[i][0] = random.gauss(0, sigma)
            v[i][1] = random.gauss(0, sigma)
            v[i][2] = random.gauss(0, sigma)

def Andersen_themrostat_2d(lmp, INCAR, PARCAR):
    nu = INCAR['nu']
    T = INCAR['T']
    dt = INCAR['dt']
    sigma = np.sqrt(T)
    v = lmp.extract_atom("v")
    N = lmp.get_natoms()
    type = lmp.extract_atom("type")
    for i in range(N):
        if (random.random() < nu * dt):
            type_i = str(type[i])
            mass = PARCAR[type_i][0]
            sigma = np.sqrt(T/mass)
            v[i][0] = random.gauss(0, sigma)
            v[i][1] = random.gauss(0, sigma)