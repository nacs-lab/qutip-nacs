"""
A Wrapper around QuTiP to support simulations of models where each site can have different number of states
"""

from os import stat
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from itertools import product

N = []
""" List of the number of states in each atom of the chain """
id_list = []
""" This variable has an identity operator as each element. """
gs_list = []
""" This variable is a ground state atom in each element. Note that the "up" state is a Rydberg atom, and "down" state is a ground state atom """
number = (int,float,complex)
""" Defines a number as an int, float or complex """
n_atoms = 0
""" The number of atoms in the chain """

qt = qt
""" The qt variable gives access to the qutip module as qt, if you'd like """

def set_n(n):
    """
    Sets the number of atoms for this module.

    Args:
        n: A list of the number of states for each atom in the chain

    Returns:
        None

    Raises:
        None

    """
    global N, id_list, gs_list, n_atoms
    N = n
    id_list = []
    gs_list = []
    for elem in n:
        id_list.append(qt.qeye(elem))
        gs_list.append(qt.basis(elem,0))
    n_atoms = len(id_list)

def create_state_list(site_idx, vals):
    """
    Creates a state_list for use in proj for instance from a list of indices and values at those indices.

    Args:
        site_idx: list of sites where we don't want identity to act
        vals: the values each of these sites should take

    Returns:
        A state_list with -1 on sites with identity and a value otherwise. 

    Raises:
        None

    """
    global N
    if isinstance(vals, number):
        vals = [vals for i in range(len(site_idx))]
    res = [-1 for i in range(len(N))]
    for idx, elem in enumerate(site_idx):
        res[elem] = vals[idx]
    return res

class States:

    @staticmethod
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

    @staticmethod
    def specify(state_list):
        """
        Returns a state with the specified states in the state_list

        Args:
            state_list: A list of the state for each atom

        Returns:
            The state which each atom in the specified state of the state_list

        Raises:
            None

        """
        global gs_list, N
        local_gs_list = gs_list.copy()
        for idx, elem in enumerate(state_list):
            if elem > 0:
                local_gs_list[idx] = qt.basis(N[idx], elem)
        return qt.tensor(*local_gs_list)

class Operators:

    @staticmethod
    def identity(bSingle = 0, idx = 0):
        """
        Returns the identity operator.

        Args:
            bSingle: boolean for a identity operator on a single site
            idx: the index of the site

        Returns:
            The identity operator.

        Raises:
            None

        """
        global id_list
        if bSingle:
            return id_list[idx]
        else:
            return qt.tensor(*id_list)

    @staticmethod
    def zero():
        """
        Returns the zero operator.

        Args:
            None

        Returns:
            The zero operator.

        Raises:
            None

        """
        global N
        space_size = np.prod(np.array(N))
        res = qt.Qobj(np.zeros((space_size, space_size)), dims=[N for i in range(2)])
        return res

    @staticmethod
    def jx(elems):
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
            local_id_list[elem] = qt.jmat((N[elem] - 1)/2, 'x')
        return qt.tensor(*local_id_list)

    @staticmethod
    def jy(elems):
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
            local_id_list[elem] = qt.jmat((N[elem] - 1)/2, 'y')
        return qt.tensor(*local_id_list)

    @staticmethod
    def jz(elems):
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
            local_id_list[elem] = qt.jmat((N[elem] - 1)/2, 'z')
        return qt.tensor(*local_id_list)

    @staticmethod
    def num(elems):
        """
        Returns an operator, where the number operator acts on the atoms at the specified indices. Otherwise, identity acts on the others.

        Args:
            elems: A list of indices on which atoms should have the number operators.

        Returns:
            An operator with number operator at the correct sites.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for elem in elems:
            local_id_list[elem] = qt.num(N[elem])
        return qt.tensor(*local_id_list)

    @staticmethod
    def sigmax(idx, states, bUnitary = 0, bSingle=0):
        """
        Returns the operator which is a spin 1/2 sigma_x on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_x operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.
            bSingle: Returns a single operator for the first element in idx and the first set of states in states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        if bSingle:
            idx = [idx[0]]
        for this_idx, elem in enumerate(idx):
            op1 = qt.basis(N[elem], states[this_idx][0]) * qt.basis(N[elem], states[this_idx][1]).dag()
            local_id_list[elem] = op1 + op1.dag()
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
        if bSingle:
            return local_id_list[idx[0]]
        else:
            return qt.tensor(*local_id_list)

    @staticmethod
    def sigmay(idx, states, bUnitary = 0, bSingle = 0):
        """
        Returns the operator which is a spin 1/2 sigma_x on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_x operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.
            bSingle: Returns a single operator for the first element in idx and the first set of states in states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        if bSingle:
            idx = [idx[0]]
        for this_idx, elem in enumerate(idx):
            op1 = 1j * qt.basis(N[elem], states[this_idx][0]) * qt.basis(N[elem], states[this_idx][1]).dag()
            local_id_list[elem] = op1 + op1.dag()
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
        if bSingle:
            return local_id_list[idx[0]]
        else:
            return qt.tensor(*local_id_list)

    @staticmethod
    def sigmaz(idx, states, bUnitary = 0, bSingle = 0):
        """
        Returns the operator which is a spin 1/2 sigma_z on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_z operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.
            bSingle: Returns a single operator for the first element in idx and the first set of states in states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        if bSingle:
            idx = [idx[0]]
        for this_idx, elem in enumerate(idx):
            op1 = qt.basis(N[elem], states[this_idx][0]).proj() - qt.basis(N[elem], states[this_idx][1]).proj()
            local_id_list[elem] = op1
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
        if bSingle:
            return local_id_list[idx[0]]
        else:
            return qt.tensor(*local_id_list)

    @staticmethod
    def proj(idx_list):
        """
        Returns the operator with the projection of each atom in the chain to a particular state. A -1 indicates identity on that site.

        Args:
            idx_list: An index for each atom to be projected into. A -1 specifies identity on this site.

        Returns:
            An appropriate projection operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for idx, elem in enumerate(idx_list):
            if elem != -1:
                local_id_list[idx] = qt.basis(N[idx], elem).proj()
        return qt.tensor(*local_id_list)

    @staticmethod
    def projOnSite(idx, state):
        """
        Returns an on-site operator for a projector.

        Args:
            idx: Site to use for the projector.
            state: State for the projector. 

        Returns:
            An appropriate projection operator.

        Raises:
            None

        """
        global N
        return qt.basis(N[idx], state).proj()

    @staticmethod
    def jump(idx_list):
        """
        Returns the operator with a jump operator on each atom in the chain. A -1 indicates identity on that site.

        Args:
            idx_list: An index for each atom to be projected into. A -1 specifies identity on this site.

        Returns:
            The appropriate jump operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for idx, elem in enumerate(idx_list):
            if elem != -1:
                local_id_list[idx] = qt.basis(N[idx], elem[0]) * qt.basis(N[idx], elem[1]).dag()
        return qt.tensor(*local_id_list)


    @staticmethod
    def PairInteraction(coords, interaction_coeff, states, atom_types = None, selection_idxs = None, powers = None):
        """
        Returns the interaction operator for atoms at particular coordinates with interaction coefficients. 

        Args:
            coords: A list of coordinates for the locations of each atom.
            interaction_coeff: A list of lists of interaction coefficients. Suppose there are 3 types of atoms, then the interaction array is \
            $ [[V_{11}, V_{12}, V_{13}], [V_{22}, V_{23}], [V_{33}]] $. A single number can also be used if there is only one type of atom.
            states: A list of the interacting state for each atom. A single number can be used for the same state on each atom.
            atom_types: A list of the atom types for each atom.
            selection_idxs: A list of the atoms involved in the interaction. By default, all atoms are involved. Note that atoms that are not selected \
            for will have their atom_types, states and coords ignored. However their entries must still be filled.
            powers: A list of lists of the exponent of the distance dependent interaction. See interaction_coeff for the format. By default, \
            the power is assumed to be -6, a van der Waals interaction.

        Returns:
            The interaction operator.

        Raises:
            None

        """
        # Add error checking
        global N
        n_atoms = len(coords)
        if isinstance(interaction_coeff, number):
            interaction_coeff = [[interaction_coeff]]
        n_types = len(interaction_coeff[0])
        if isinstance(states, number):
            states = [states for i in range(n_atoms)]
        if atom_types is None:
            atom_types = [0 for i in range(n_atoms)]
        if powers is None:
            powers = []
            for i in range(n_types):
                this_list = []
                for j in range(n_types - i):
                    this_list.append(-6)
                powers.append(this_list)
        if selection_idxs is None:
            selection_idxs = [i for i in range(n_atoms)]
        if isinstance(powers, int):
            prev_power = powers
            powers = []
            for i in range(n_types):
                this_list = []
                for j in range(n_types - i):
                    this_list.append(prev_power)
                powers.append(this_list)

        space_size = np.prod(np.array(N))
        proj_specifier = [-1 for i in range(n_atoms)]
        res = qt.Qobj(np.zeros((space_size, space_size)), dims=[N for i in range(2)])
        for i in range(n_atoms):
            if i not in selection_idxs:
                continue
            for j in range(i):
                if j not in selection_idxs:
                    continue
                pos1 = coords[i]
                pos2 = coords[j]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                type1 = min(atom_types[j],atom_types[i])
                type2 = max(atom_types[j],atom_types[i])
                inter_strength = interaction_coeff[type1][type2 - type1] * dist**(powers[type1][type2 - type1])
                this_proj_specifier = proj_specifier.copy()
                this_proj_specifier[i] = states[i]
                this_proj_specifier[j] = states[j]
                res = res + inter_strength * Operators.proj(this_proj_specifier)
        return res

    @staticmethod
    def cGate(bit_type, target_op):
        """
            Returns a controlled gate.

            Args:
                bit_type: A list of values for each atom. A number greater than 0 indicates that atom is a control atom, and will \
                induce the operation when the atom is in the specified state. For instance 3 indicates the atom induces the operation \
                when in state 3. -1 indicates that the atom does not participate in the gate. -2 indicates the atom is a target. \
                -2 corresponds to the first element of target_op, -3 indicates the next element and so on.
                target_op: A list of target operations. The first target operation is addressed with a bit_type of -2. The next is -3 and so on.

            Returns:
                The controlled gate.

            Raises:
                None

        """
        global id_list, N
        if not isinstance(target_op, list):
            target_op = [target_op]
        control_list = id_list.copy()
        op1 = id_list.copy()
        for idx, elem in enumerate(bit_type):
            if elem >= 0:
                # control bit, controlling when projector into elem.
                control_list[idx] = qt.basis(N[idx], elem).proj()
                op1[idx] = qt.basis(N[idx], elem).proj()
            elif elem <= -2:
                op1[idx] = target_op[-elem - 2]
        return qt.tensor(*op1) + qt.tensor(*id_list) - qt.tensor(*control_list)

    @staticmethod
    def specify(op_list):
        """
        Returns the operator with the operators specified on each atom in the chain.

        Args:
            op_list: A list corresponding to an operator on each site. A 1 indicates identity on that site, and a -1 is negative identity on that site.

        Returns:
            The appropriate operator.

        Raises:
            None

        """
        global id_list
        local_id_list = id_list.copy()
        for idx, elem in enumerate(op_list):
            if elem == 1:
                continue
            elif elem == -1:
                local_id_list[idx] = -local_id_list[idx]
            else:
                local_id_list[idx] = elem
        return qt.tensor(*local_id_list)


class Dynamics:
    def Unitary(H, init_state, tend, n_points, expects = [], names = [], bPlot = False):
        """
        Runs a unitary simulation (`qutip.sesolve`) and returns a `qutip.solver.Result` object. 

        Args:
            H: A Hamiltonian to run the simulation with.
            init_state: An initial state
            tend: The time to run the simulation to.
            n_points: The number of time points for the simulation. They are evenly distributed from t = 0 to tend
            expects: A list of expectation values to calculate for each time point.
            names: The names for each expectation value, which is used when plotting.
            bPlot: Whether to plot the result. 

        Returns:
            A `qutip.solver.Result` object.

        Raises:
            None

        """
        if isinstance(init_state, qt.qobj.Qobj):
            init = init_state
        else:
            init = States.specify(init_state)
        tlist = np.linspace(0, tend, n_points)
        e_list = []
        e_str = []
        for idx, elem in enumerate(expects):
            if isinstance(elem, list):
                e_list.append(Operators.proj(elem))
                if names == []:
                    e_str.append(str(elem))
                else:
                    e_str.append(names[idx])
            else:
                e_list.append(elem)
                if names == []:
                    e_str.append('qutip operator')
                else:
                    e_str.append(names[idx])
            
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

    def MasterEq(H, init_state, tend, n_points, jump_ops = [], expects = [], names = [], bPlot = False):
        """
        Runs a master equation simulation (`qutip.mesolve`) and returns a `qutip.solver.Result` object. 

        Args:
            H: A Hamiltonian to run the simulation with.
            init_state: An initial state
            tend: The time to run the simulation to.
            n_points: The number of time points for the simulation. They are evenly distributed from t = 0 to tend
            jump_ops: A list of jump operators to use in the simulation
            expects: A list of expectation values to calculate for each time point.
            names: The names for each expectation value, which is used when plotting.
            bPlot: Whether to plot the result. 

        Returns:
            A `qutip.solver.Result` object.

        Raises:
            None

        """
        if isinstance(init_state, qt.qobj.Qobj):
            init = init_state
        else:
            init = States.specify(init_state)
        tlist = np.linspace(0, tend, n_points)
        e_list = []
        e_str = []
        for idx, elem in enumerate(expects):
            if isinstance(elem, list):
                e_list.append(Operators.proj(elem))
                if names == []:
                    e_str.append(str(elem))
                else:
                    e_str.append(names[idx])
            else:
                e_list.append(elem)
                if names == []:
                    e_str.append('qutip operator')
                else:
                    e_str.append(names[idx])
            
        opts = qt.solver.Options(store_final_state=True)
        res = qt.mesolve(H, init, tlist, jump_ops, e_list, options=opts)
        if bPlot:
            n_expects = len(e_list)
            plt.figure()
            for i in range(n_expects):
                plt.plot(tlist, res.expect[i])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend(e_str)
        return res

def CalculateFidelity(func, target_op, states = None):
    global N
    if states is None:
        states = []
        for elem in N:
            states.append([i for i in range(elem)])
    nstates_calculated = 1
    for elem in states:
        nstates_calculated = nstates_calculated * len(elem)
    nstates = np.prod(np.array(N))
    result_matrix = np.zeros((nstates, nstates), dtype=np.complex128)
    ideal_result_matrix = np.zeros((nstates, nstates), dtype=np.complex128)
    op_dims = [N for i in range(2)]
    op_shape = (nstates, nstates)
    idx = 0
    for perm in product(*states):
        init_state = States.specify(perm)
        result_state = func(init_state)
        result_matrix[:,idx] = np.squeeze(result_state.full())
        ideal_result_state = target_op * init_state
        ideal_result_matrix[:, idx] = np.squeeze(ideal_result_state.full())
        idx = idx + 1
    result_op = qt.Qobj(result_matrix, dims=op_dims, shape=op_shape)
    ideal_result_op = qt.Qobj(ideal_result_matrix, dims=op_dims, shape=op_shape)
    return (result_op.dag() * ideal_result_op).tr() / nstates_calculated, result_op, ideal_result_op