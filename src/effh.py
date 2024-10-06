import numpy as np

class EffectiveHamiltonian_class:
    def __init__(self, He, Ha, Llist, LdL):
        self.He = He  # Hermitian part
        self.Ha = Ha  # Anti-Hermitian part
        self.Llist = Llist  # List of Lindblad operators
        self.LdL = LdL  # List of L†L

def EffectiveHamiltonian( mats, Llist):
    """
    Create an EffectiveHamiltonian object based on provided parameters.

    :param funcs: List of time-dependent functions for the Hamiltonian.
    :param mats: List of matrices (Hamiltonian terms).
    :param γlist: List of coefficients for each Lindblad operator set.
    :param Llist: List of lists of Lindblad operators.
    :return: An instance of EffectiveHamiltonian_class.
    """
    # Directly use the input matrices as the Hermitian part (assuming they are Hermitian)
    He = sum(mats)  # Sum of Hamiltonian terms as Hermitian part
    Ha = 0.0  # Initialize anti-Hermitian part
    LdL = []  # Initialize the list for Lindblad operator products

    # Combine γ and Lindblad operators (L * L')
    for  LL in Llist:
        for L in LL:
            # print("L in effh:",np.shape(L))
            L_dagger_L = (L.conj().T @ L)  # Compute γ * L†L
            LdL.append(L_dagger_L)  # Append to LdL list
            Ha += L_dagger_L  # Sum for the anti-Hermitian part
            # print(Ha)

    # Return the Effective Hamiltonian object
    return EffectiveHamiltonian_class(He, 0.5 * Ha, [L for LL in Llist for L in LL], LdL)

# Helper methods that align with the Julia logic
def herm(H, t):
    return H.He(t)  # Assuming He is callable with t

def antiherm(H):
    return H.Ha

def get_LdL(H):
    return H.LdL

def get_L(H):
    return H.Llist
