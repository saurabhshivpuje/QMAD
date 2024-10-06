import numpy as np
from itertools import combinations, product
from scipy.linalg import expm, kron
from numpy.random import rand
random_numbers = np.random.uniform(low=10000, high=99999)
np.random.seed(int(random_numbers))
from ansatz import *


# Helper Functions for evolution
# def tag(A):
#     return A.tag if A else None #!!!!!!!a.any or a.all??
def tag(A):
    return getattr(A, 'tag', None) if A is not None else None


def aexp(A, theta):
    return expm(-1j * theta * A.mat / 2)

def add_A(A, P):
    A.A.append(P)
    A.theta = np.append(A.theta, 0)

def lmul(A, ψ):
    # print("ehrerw:",A)
    # print("amat:",np.shape(A.mat))
    # print("her in lmul",A.mat @ ψ)
    return A.mat @ ψ

def update_theta(ansatz, dtheta, dt):
    ansatz.theta = ansatz.theta + dtheta * dt

def partial_theta(A):
    len_A = len(A.A)
    ψ = A.ref.copy()
    res = []
    for i in range(len_A):
        ψ = aexp(A.A[i], A.theta[i]) @ ψ
        ψt = -0.5j * lmul(A.A[i], ψ)
        for j in range(i + 1, len_A):
            ψt = aexp(A.A[j], A.theta[j]) @ ψt
        res.append(ψt)
    return res

def build_m(ψ, dψ):
    l = len(dψ)
    res = np.zeros((l, l))
    ψ = ψ[:, np.newaxis] if len(ψ.shape) == 1 else ψ

    for μ in range(l):
        for ν in range(l):
            res[μ, ν] = 2 * np.real(dψ[μ].T.conj() @ dψ[ν] + ψ.T.conj() @ dψ[μ] @ ψ.T.conj() @ dψ[ν])
    return res

def update_m(m, dψa, ψ, dψ):
    l = m.shape[0]
    mp = np.zeros((l + 1, l + 1))
    dψa = dψa[:, np.newaxis]
    ψ = ψ[:, np.newaxis]
    mp[l, l] = 2 * np.real((dψa.T.conj() @ dψa + ψ.T.conj() @ dψa @ ψ.T.conj() @ dψa).item())
    mp[:l, :l] = m
    for μ in range(l):
        temp = 2 * np.real(dψ[μ].T.conj() @ dψa + ψ.T.conj() @ dψ[μ] @ ψ.T.conj() @ dψa)
        mp[μ, l] = temp.item()
        mp[l, μ] = temp.item()
    return mp

def build_v(ψ, dψ, He, Ha):
    L = -1j * (He - 1j * Ha)
    l = len(dψ)
    ψ = ψ[:, np.newaxis] if len(ψ.shape) == 1 else ψ
    res = np.zeros(l)
    for μ in range(l):
        res[μ] = 2 * np.real((dψ[μ].T.conj() @ L @ ψ + dψ[μ].T.conj() @ ψ @ ψ.T.conj() @ L.T.conj() @ ψ).item())
    return res

def update_v(v, dψₐ, ψ, He, Ha):
    L = -1j * (He - 1j * Ha)
    l = len(v)
    dψₐ = dψₐ[:, np.newaxis]
    ψ = ψ[:, np.newaxis]
    vp = np.zeros(l + 1)
    vp[:l] = v
    vp[l] = 2 * np.real((dψₐ.T.conj() @ L @ ψ + dψₐ.T.conj() @ ψ @ ψ.T.conj() @ L.T.conj() @ ψ).item())
    return vp

def lin_solve(m, v, λ=1e-4):
    A = m.T @ m
    try:
        xλ = np.linalg.solve(A + λ**2 * np.eye(A.shape[0]), m.T @ v)
    except np.linalg.LinAlgError:
        xλ = np.linalg.lstsq(A + λ**2 * np.eye(A.shape[0]), m.T @ v, rcond=None)[0]
    return xλ, 2 * v.T @ xλ - xλ.T @ m @ xλ

def get_newest_A(A):
    return A.A[-1] if A.A else None

def update_state(A):
    if A.A:
        state = A.ref.copy()
        for i in range(len(A.A)):
            state = aexp(A.A[i], A.theta[i]) @ state
        A.state = state
    else:
        A.state = A.ref

def set_ref(A, ref):
    A.ref = ref

def reset(A):
    A.theta = np.array([])
    A.A = []
    update_state(A)

class Result:
    def __init__(self, t, psi, energy, pop, theta, A, jump_t, jump_L, status):
        self.t = t
        self.psi = psi
        self.energy = energy  # Add energy attribute
        self.pop = pop
        self.theta = theta
        self.A = A
        self.jump_t = jump_t
        self.jump_L = jump_L
        self.status = status

    def __repr__(self):
        return f"Result(t={self.t}, psi={self.psi}, energy={self.energy}, pop={self.pop}, theta={self.theta}, A={self.A}, jump_t={self.jump_t}, jump_L={self.jump_L}, status={self.status})"

class AVQDSol:
    def __init__(self, t, u, θ, A, jump_L, jump_t, status=0, norm=[]):
        self.t = t
        self.u = u
        self.θ = θ
        self.A = A
        self.jump_L = jump_L
        self.jump_t = jump_t
        self.status = status
        self.norm = norm

def one_step(A, He, Ha, dt):

    relrcut = A.relrcut
    psi_ = A.state
    dpsi_ = partial_theta(A)
    M = build_m(psi_, dpsi_)
    V = build_v(psi_, dpsi_, He, Ha)
    dtheta, vmv = lin_solve(M, V)
    vmvMax = vmv
    Mtmp = M
    Vtmp = V
    dthetatmp = dtheta
    opTmp = None
    dpsi_ₐTmp = None
    add_flag = True

    while add_flag:
        # print("here in onestep While")
        tagTmp = None
        # print("Ansatz pool:",A.pool)
        for op in A.pool:
            # print("opertaro from pool:",(op.tag))
            if tag(get_newest_A(A)) == tag(op):
                # print("here in get newest_A")
                continue
            # print("here in op for loop")
            dpsi_ₐ = -0.5j * lmul(op, psi_)

            Mop = update_m(M, dpsi_ₐ, psi_, dpsi_)
            # print("Mop:",Mop)
            Vop = update_v(V, dpsi_ₐ, psi_, He, Ha)
            dthetaop, vmvOp = lin_solve(Mop, Vop)
            # print("berfore the if:","vmvOp:",vmvOp, "vmvMax:",vmvMax)

            if vmvOp > vmvMax:
                # print("vmvOp:",vmvOp, "vmvMax:",vmvMax)

                Mtmp = Mop
                Vtmp = Vop
                vmvMax = vmvOp
                dthetatmp = dthetaop
                opTmp = op
                tagTmp = tag(op)
                # print("tagTmp:",tagTmp)
                dpsi_ₐTmp = dpsi_ₐ
        # print("vmvMax - vmv:",vmvMax-vmv)
        add_flag = vmvMax - vmv >= relrcut

        

        if tagTmp is not None and add_flag:

            add_A(A, opTmp)
            vmv = vmvMax
            M = Mtmp
            V = Vtmp
            dtheta = dthetatmp
            dpsi_.append(dpsi_ₐTmp)

    update_theta(A, dtheta, dt)
    # print("Ansatz:",A.theta)
    update_state(A)


def solve_avq(H, A, tspan, dt, save_state=True, save_everystep=True):
    # Store initial reference state for reinitialization
    ref_init = A.ref.copy()
    update_state(A)
    
    # Initialize time and recorder lists
    t = tspan[0]
    t_list = []
    u_list = []
    theta_list = []
    A_list = []
    jump_L_list = []
    jump_t_list = []

    # Break flag for error handling
    break_flag = False
    Γ = 0  # Jump recorder
    q = rand()  # Random number for quantum jump
    # print("expo:",np.exp(-Γ))
    # print("random q:", q)

    if save_everystep:
        t_list.append(t)
        theta_list.append(A.theta.copy())

        A_list.append([tag(a) for a in A.A])
        if save_state:
            u_list.append(A.state.copy())

    # Main evolution loop
    while t + dt <= tspan[1]:
        print("timestep:",dt)
        He = H.He
        # He=np.zeros((32,32))
        # print("Hermitian ham:",np.shape(He))
        Ha = H.Ha
        # Ha=np.zeros((32,32))

        # print("antihermitian ham:",np.shape(Ha))
        # stop
        # Perform the time evolution if no quantum jump is needed
        # print("random number:",q)
        if np.exp(-Γ) > q:
            # print("here not in jump")
            try:
                # print("here0",A)
                one_step(A, He, Ha, dt)
                # print("here")
            except Exception:
                break_flag = True
                break
            psi_ = A.state
            # print("psi_Tconj",np.shape(psi_.T.conj()))
            # print("Ha",np.shape(Ha))
            # print("psi_",np.shape(psi_))
            # stop
            Γ += 2 * np.real(psi_.T.conj() @ Ha @ psi_) * dt
            # print("new gamma:",Γ)

            if save_everystep:
                t_list.append(t + dt)
                # print("tlist append:",t_list)
                theta_list.append(A.theta.copy())
                A_list.append([tag(a) for a in A.A])

        else:
            # Apply quantum jump
            # print("JUMPPPP!")
            psi_ = A.state

            weights = np.array([np.real(psi_.T.conj() @ LdL @ psi_) for LdL in H.LdL])
            # print("llist shpae",type(H.Llist))
            # print("lsiting",np.shape([matrix for matrix in H.Llist]))
            # L_index = np.random.choice([i for i in range (0,np.shape(H.Llist)[0])], p=weights/weights.sum())#!!!!!!!!!!!]
            # print("Lindex:", L_index)
            L_index=np.random.choice([i for i in range (0,np.shape(H.Llist)[0])],p=weights/weights.sum())
            L=H.Llist[L_index]
            # print("Lindex:",L_index)
            # print("here")
            # stop
            # print("shape:",np.shape(L))
            psi_n = L @ psi_ / np.linalg.norm(L @ psi_)

            # Record jump information
            jump_t_list.append(t)
            jump_L_list.append(tag(L))

            t_list.append(t + dt)
            set_ref(A, psi_n)
            reset(A)

            # Record ansatz parameters post-jump
            theta_list.append(A.theta.copy())
            A_list.append([tag(a) for a in A.A])

            # Reset jump recorder
            Γ = 0
            q = rand()

        # Save state and increment time
        if save_state:
            u_list.append(A.state.copy())
        t += dt

    # Final check for status if no errors occurred
    if not break_flag:
        if (not jump_t_list or not np.isclose(t, jump_t_list[-1])) and not save_everystep:
            theta_list.append(A.theta.copy())
            A_list.append([tag(a) for a in A.A])

    # Reset the ansatz to the initial condition
    # print("ansatz before reset:",A.A)
    set_ref(A, ref_init)
    reset(A)
    # print("tlist:",t_list)
    # Return solution object
    return AVQDSol(t_list, u_list, theta_list, A_list, jump_L_list, jump_t_list, status=0 if not break_flag else 1)

# Function to evolve a single trajectory and collect results
def solve_avq_trajectory(H, ansatz, tf, dt):
    # Solve for a single trajectory using solve_avq
    res = solve_avq(H, ansatz, [0, tf], dt, save_state=True, save_everystep=True)
    
    # Post-process the results to extract energy and populations
    # print("time:",res.t)
    tlist = res.t
    psi_list = res.u
    energy = []
    pop = []
    
    for i in range(len(tlist)):
        psi = psi_list[i]
        energy.append(np.real(np.conj(psi.T) @ (H.He) @ psi))  # Energy
        # print("pop1:",pop)
        pop.append([np.abs(psi[0])**2, np.abs(psi[1])**2])  # Population in ground/excited states for amplitude damping
        # print("pop2:",pop)

    return Result(tlist, psi_list, energy, pop, res.θ, res.A, res.jump_t, res.jump_L, res.status)
    pass