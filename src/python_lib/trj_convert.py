import numpy as np

def trj_converter(dic_name, file_name, data_dir):

    with open(dic_name + file_name + ".lammpstrj", "r") as f:
        i = 0
        data = f.readlines()
        for line in data:
            data[i] = line.strip('\n')
            i += 1


        total_line_num = np.shape(data)[0]
        count_flag = 0
        end_flag = 1
        particle_num = int(data[3])
        single_line_num = particle_num + 9
        steps = int(total_line_num / single_line_num)
        pos_data = np.zeros([steps, particle_num, 5])
        step_data = np.zeros(steps)

        for i in range(steps):
            step = int(data[i * single_line_num + 1])
            step_data[i] = step
            start = i * single_line_num + 9
            end = (i + 1) * single_line_num
            h = 0
            for k in range(start, end):
                info = data[k]
                info = info.split( )
                for j in range(2):
                    info[j] = int(info[j])
                info = np.array(info)
                pos_data[i][h] = info
                h += 1
        np.save(data_dir + file_name, pos_data)

def POS_convert(file_name):
    data = np.loadtxt(file_name)
    M = np.shape(data)[0]
    POSCAR = np.zeros((M, 4))
    for i in range(M):
        id, q_l, x, y, z = data[i]
        q = 2 * (q_l - 1.5)
        POSCAR[i] = q, x, y, z
    np.savetxt('POSCAR_' + file_name, POSCAR)

    return 'success'

def POS_convert_direct(data):
    M = np.shape(data)[0]
    POSCAR = np.zeros((M, 4))
    for i in range(M):
        id, q_l, x, y, z = data[i]
        q = 2 * (q_l - 1.5)
        POSCAR[i] = q, x, y, z
    return POSCAR