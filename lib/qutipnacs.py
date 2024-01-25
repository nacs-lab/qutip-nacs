import qutip as qt
import numpy as np
from matplotlib import pyplot as plt

N = 0
id_list = []
gs_list = [] # list of ground state 
number = (int,float,complex)

qt = qt

def set_n(n):
    global N, id_list, gs_list
    N = n
    id_list = [qt.qeye(2) for i in range(n)]
    gs_list = [qt.basis(2,1) for i in range(n)]

def all_ground():
    global gs_list
    return qt.tensor(*gs_list)

def up_state(up_list):
    global gs_list
    local_gs_list = gs_list.copy()
    for elem in up_list:
        local_gs_list[elem] = qt.basis(2,0)
    return qt.tensor(*local_gs_list)

def identity():
    global id_list
    return qt.tensor(*id_list)

def sigmax(elems):
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmax()
    return qt.tensor(*local_id_list)

def sigmay(elems):
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmay()
    return qt.tensor(*local_id_list)

def sigmaz(elems):
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmaz()
    return qt.tensor(*local_id_list)

def n(elems):
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = (1 + qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def g(elems):
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = (1 - qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def occ_op(occ_list):
    global id_list
    local_id_list = id_list.copy()
    for elem in occ_list:
        if elem == 1:
            # up state
            local_id_list[elem] = (1 + qt.sigmaz())/2
        else:
            local_id_list[elem] = (1 - qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def PairInteraction(coords, interaction_coeff, atom_types = None, powers = None):
    # Add error checking
    n_atoms = len(coords)
    if isinstance(interaction_coeff, number):
        interaction_coeff = [[interaction_coeff]]
    n_types = len(interaction_coeff[0])
    if atom_types is None:
        atom_types = [0 for i in range(n_atoms)]
    if powers is None:
        powers = []
        for i in range(n_types):
            this_list = []
            for j in range(n_types - i):
                this_list.append(-6)
            powers.append(this_list)
    if isinstance(powers, int):
        prev_power = powers
        powers = []
        for i in range(n_types):
            this_list = []
            for j in range(n_types - i):
                this_list.append(prev_power)
            powers.append(this_list)

    res = qt.Qobj(np.zeros((2**n_atoms, 2**n_atoms)), dims=[[2 for i in range(n_atoms)], [2 for i in range(n_atoms)]])
    for i in range(n_atoms):
        for j in range(i):
            pos1 = coords[i]
            pos2 = coords[j]
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            type1 = min(atom_types[j],atom_types[i])
            type2 = max(atom_types[j],atom_types[i])
            inter_strength = interaction_coeff[type1][type2 - type1] * dist**(powers[type1][type2 - type1])
            res = res + inter_strength * n([i,j])
    return res

def DrivenRyd(drive, det, coords, interaction_coeff, atom_types = None, powers = None):
    n_atoms = len(coords)
    if isinstance(drive, number):
        drive = [drive for i in range(n_atoms)]
    if isinstance(det, number):
        det = [det for i in range(n_atoms)]
    HTot = PairInteraction(coords, interaction_coeff, atom_types, powers)
    for i in range(n_atoms):
        HTot = HTot + drive[i] * sigmax([i]) / 2 - det[i] * n([i])
    return HTot

def time_indep_unitary(H, init_state, tend, n_points, expects, bPlot = False):
    init = up_state(init_state)
    tlist = np.linspace(0, tend, n_points)
    e_list = []
    e_str = []
    for elem in expects:
        e_list.append(occ_op(elem))
        e_str.append(str(elem))
    res = qt.sesolve(H, init, tlist, e_list)
    if bPlot:
        n_expects = len(e_list)
        plt.figure()
        for i in range(n_expects):
            plt.plot(tlist, res.expect[i])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(e_str)
    return res



