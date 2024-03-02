"""
A Wrapper around QuTiP to support simulations of spin-like models, in particular the Rydberg model.
"""

import qutip as qt
import numpy as np
from matplotlib import pyplot as plt

N = 0
""" The number of atoms in the current instance of the module """
id_list = []
""" This variable has an identity operator as each element. """
gs_list = []
""" This variable is a ground state atom in each element. Note that the "up" state is a Rydberg atom, and "down" state is a ground state atom """
number = (int,float,complex)
""" Defines a number as an int, float or complex """

qt = qt
""" The qt variable gives access to the qutip module as qutipnacs.qt, if you'd like """

def set_n(n):
    """
    Sets the number of atoms for this module.

    Args:
        n: The number of atoms in this module.

    Returns:
        None

    Raises:
        None

    """
    global N, id_list, gs_list
    N = n
    id_list = [qt.qeye(2) for i in range(n)]
    gs_list = [qt.basis(2,1) for i in range(n)]

def all_ground():
    """
    Returns the all ground state.

    Args:
        None:

    Returns:
        The all ground state. 

    Raises:
        None

    """
    global gs_list
    return qt.tensor(*gs_list)

def up_state(up_list):
    """
    Returns a state with an atom in the up state specified by index in the up_list.

    Args:
        up_list: A list of indices where an atom should be in the up state

    Returns:
        The resulting state.

    Raises:
        None

    """
    global gs_list
    local_gs_list = gs_list.copy()
    for elem in up_list:
        local_gs_list[elem] = qt.basis(2,0)
    return qt.tensor(*local_gs_list)

def identity():
    """
    Returns the identity operator.

    Args:
        None

    Returns:
        The identity operator.

    Raises:
        None

    """
    global id_list
    return qt.tensor(*id_list)

def sigmax(elems):
    """
    Returns an operator, where $ \\sigma_x $ acts on the atoms at the specified indices. Otherwise, identity acts on the others.

    Args:
        elems: A list of indices on which atoms should have pauli X acting on it.

    Returns:
        An operator with $ \\sigma_x $ at the correct sites.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmax()
    return qt.tensor(*local_id_list)

def sigmay(elems):
    """
    Returns an operator, where $ \\sigma_y $ acts on the atoms at the specified indices. Otherwise, identity acts on the others.

    Args:
        elems: A list of indices on which atoms should have pauli Y acting on it.

    Returns:
        An operator with $ \\sigma_y $ at the correct sites.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmay()
    return qt.tensor(*local_id_list)

def sigmaz(elems):
    """
    Returns an operator, where $ \\sigma_z $ acts on the atoms at the specified indices. Otherwise, identity acts on the others.

    Args:
        elems: A list of indices on which atoms should have pauli Z acting on it.

    Returns:
        An operator with $ \\sigma_z $ at the correct sites.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = qt.sigmaz()
    return qt.tensor(*local_id_list)

def n(elems):
    """
    Returns an operator, where the Rydberg number operator acts on the atoms at the specified indices. Otherwise, identity acts on the others.

    Args:
        elems: A list of indices on which atoms should have rydberg occupation acting on it.

    Returns:
        An operator with Rydberg occupation at the correct sites.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = (1 + qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def g(elems):
    """
    Returns an operator, where ground state occupation operator acts on the atoms at the specified indices. Otherwise, identity acts on the others.

    Args:
        elems: A list of indices on which atoms should have ground state occupation acting on it.

    Returns:
        An operator with ground state occupation at the correct sites.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for elem in elems:
        local_id_list[elem] = (1 - qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def occ_op(occ_list):
    """
    Returns an operator, which projects out a particular bit string (of 0s and 1s) on the set of sites. 0 corresponds to ground state, 1 corresponds to Rydberg state.

    Args:
        elems: A list of 0s and 1s indicating the state of each atom in this operator.

    Returns:
        An operator with the correct ground state and rydberg atom operators.

    Raises:
        None

    """
    global id_list
    local_id_list = id_list.copy()
    for idx, elem in enumerate(occ_list):
        if elem == 1:
            # up state
            local_id_list[idx] = (1 + qt.sigmaz())/2
        else:
            local_id_list[idx] = (1 - qt.sigmaz())/2
    return qt.tensor(*local_id_list)

def PairInteraction(coords, interaction_coeff, atom_types = None, powers = None):
    """
    Returns the interaction operator for atoms at particular coordinates with interaction coefficients. 

    Args:
        coords: A list of coordinates for the locations of each atom.
        interaction_coeff: A list of lists of interaction coefficients. Suppose there are 3 types of atoms, then the interaction array is
        $ [[V_{11}, V_{12}, V_{13}], [V_{22}, V_{23}], [V_{33}]] $. A single number can also be used if there is only one type of atom.
        atom_types: A list of the atom types for each atom.
        powers: A list of lists of the exponent of the distance dependent interaction. See interaction_coeff for the format. By default,
        the power is assumed to be -6, a van der Waals interaction.

    Returns:
        The interaction operator.

    Raises:
        None

    """
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
    if isinstance(init_state, qt.qobj.Qobj):
        init = init_state
    else:
        init = up_state(init_state)
    tlist = np.linspace(0, tend, n_points)
    e_list = []
    e_str = []
    for elem in expects:
        e_list.append(occ_op(elem))
        e_str.append(str(elem))
    opts = qt.solver.Options(store_final_state=True)
    res = qt.sesolve(H, init, tlist, e_list, options=opts)
    if bPlot:
        n_expects = len(e_list)
        plt.figure()
        for i in range(n_expects):
            plt.plot(tlist, res.expect[i])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(e_str)
    return res

def cGate(bit_type, target_op):
    global id_list
    control_list = id_list.copy()
    op1 = id_list.copy()
    for idx, elem in enumerate(bit_type):
        if elem == 0:
            # control bit, controlling when down.
            control_list[idx] = (1 - qt.sigmaz())/2
            op1[idx] = (1 - qt.sigmaz())/2
        elif elem == 1:
            control_list[idx] = (1 + qt.sigmaz())/2
            op1[idx] = (1 + qt.sigmaz())/2
        elif elem == 2:
            op1[idx] = target_op
    return qt.tensor(*op1) + qt.tensor(*id_list) - qt.tensor(*control_list)

def get_fidelity(func, target_op):
    # Iterate over all initial states
    global N, gs_list, id_list
    gs_state = qt.tensor(*gs_list)
    id_op = qt.tensor(*id_list)
    state_dims = gs_state.dims
    state_shape = gs_state.shape
    op_dims = id_op.dims
    op_shape = id_op.shape
    result_matrix = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(2**N):
        init_np_state = np.zeros(2**N)
        init_np_state[i] = 1.0
        init_state = qt.Qobj(init_np_state, dims = state_dims, shape=state_shape)
        result_state = func(init_state)
        result_matrix[:,i] = np.squeeze(result_state.full())
    result_op = qt.Qobj(result_matrix, dims=op_dims, shape=op_shape)
    return (result_op.dag() * target_op).tr() / 2**N, result_op