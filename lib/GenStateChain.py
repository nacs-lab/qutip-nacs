"""
A Wrapper around QuTiP to support simulations of models where each site can have different number of states
"""

from os import stat
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt

N = []
""" List of the number of states in each atom of the chain """
id_list = []
""" This variable has an identity operator as each element. """
gs_list = []
""" This variable is a ground state atom in each element. Note that the "up" state is a Rydberg atom, and "down" state is a ground state atom """
number = (int,float,complex)
""" Defines a number as an int, float or complex """

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
    global N, id_list, gs_list
    N = n
    id_list = []
    gs_list = []
    for elem in n:
        id_list.append(qt.qeye(elem))
        gs_list.append(qt.basis(elem,0))

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
    def sigmax(idx, states, bUnitary = 0):
        """
        Returns the operator which is a spin 1/2 sigma_x on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_x operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for this_idx, elem in enumerate(idx):
            op1 = qt.basis(N[elem], states[this_idx][0]) * qt.basis(N[elem], states[this_idx][1]).dag()
            local_id_list[elem] = op1 + op1.dag()
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
        return qt.tensor(*local_id_list)

    @staticmethod
    def sigmay(idx, states, bUnitary = 0):
        """
        Returns the operator which is a spin 1/2 sigma_x on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_x operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for this_idx, elem in enumerate(idx):
            op1 = 1j * qt.basis(N[elem], states[this_idx][0]) * qt.basis(N[elem], states[this_idx][1]).dag()
            local_id_list[elem] = op1 + op1.dag()
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
        return qt.tensor(*local_id_list)

    @staticmethod
    def sigmaz(idx, states, bUnitary = 0):
        """
        Returns the operator which is a spin 1/2 sigma_z on the list of idxs. The states to swap are specified in states.

        Args:
            idx: A list of indices for where we want to have a spin 1/2 sigma_z operator
            states: A list of lists, where each member is the states to swap for that particular site.
            bUnitary: Set to 1 to get an identity on all other states of the affected idx. Otherwise, the operator will annihilate unused states.

        Returns:
            The appropriate swap operator.

        Raises:
            None

        """
        global id_list, N
        local_id_list = id_list.copy()
        for this_idx, elem in enumerate(idx):
            op1 = qt.basis(N[elem], states[this_idx][0]).proj() + qt.basis(N[elem], states[this_idx][1]).proj()
            local_id_list[elem] = op1
            if bUnitary:
                for i in range(N[elem]):
                    if i not in states[this_idx]:
                        local_id_list[elem] = local_id_list[elem] + qt.basis(N[elem], i).proj()
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
    def PairInteraction(coords, interaction_coeff, states, atom_types = None, selection_idxs = None, powers = None):
        """
        Returns the interaction operator for atoms at particular coordinates with interaction coefficients. 

        Args:
            coords: A list of coordinates for the locations of each atom.
            interaction_coeff: A list of lists of interaction coefficients. Suppose there are 3 types of atoms, then the interaction array is
            $ [[V_{11}, V_{12}, V_{13}], [V_{22}, V_{23}], [V_{33}]] $. A single number can also be used if there is only one type of atom.
            states: A list of the interacting state for each atom. A single number can be used for the same state on each atom.
            atom_types: A list of the atom types for each atom.
            selection_idxs: A list of the atoms involved in the interaction. By default, all atoms are involved. Note that atoms that are not selected
            for will have their atom_types, states and coords ignored. However their entries must still be filled.
            powers: A list of lists of the exponent of the distance dependent interaction. See interaction_coeff for the format. By default,
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