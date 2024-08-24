No_spins = 3
N = No_spins
No_exc = No_spins
spin_type = 1 / 2
spin_geom = "chain"
lat_const = 0.1
transition_omega = 2 * pi
Spin_system_object = spin_object_constructor(No_spins, No_exc, spin_type, transition_omega, spin_geom, lat_const);   #M is now disregarded
Spin_system_object = Ham_and_state_constructor(Spin_system_object)
Spin_system_object = diagonal_jump_operators(Spin_system_object)

H = Matrix(Spin_system_object.H_eff.data)

tf = 3.0
dt = 0.001
u0 = zeros(ComplexF64, 2^N, 2^N) #[1.0+0im 0 0 0;0 0 0 0; 0 0 0 0; 0 0 0 0]
u0[1, 1] = 1.0 + 0im

H = Hamiltonian([(s) -> 1.0], [H], unit=:ħ)

Gamma = Matrix(Spin_system_object.Gamma_mat)
γ = Spin_system_object.gamma_diag
L = [Matrix(Spin_system_object.J_diag[i].data) for i in 1:N]
linds = [Lindblad(γ[i], L[i]) for i in 1:N]
linds = InteractionSet(linds...)

annealing = Annealing(H, u0, interactions=linds)
sol = solve_lindblad(annealing, tf, alg=Tsit5(), abstol=1e-6, reltol=1e-6, saveat=0:dt:tf)
# jldsave("dicke_exact_0.1.jld2"; sol.u)

steps=Integer(tf/dt)
excited=[Float64(real(sol.u[i][1])) for i in 1:steps]
# ground=[Float64(real(sol.u[i][2^(2N)])) for i in 1:steps]

# times = LinRange(0, tf, steps)
# Plots.plot(times, excited, label="Excited State", xlabel="Time(s)", ylabel="Amplitude", linewidth=2)
# plot!(times, ground, label="Ground State", linewidth=2)

steps = Integer(tf / dt)

Spin_obj = Spin_system_object
rho = sol.u
prop_emiss = zeros(Float64, steps)
prop_Pop_ee = zeros(Float64, steps)

for j in 1:steps
    prop_emiss[j] = real(tr(sum((Spin_obj.gamma_diag[i]) * (dagger(Spin_obj.J_diag[i]) * Spin_obj.J_diag[i]) for i = 1:N).data * sol.u[j]))
    prop_Pop_ee[j] = real(tr(sum([spin_collection_op(Spin_obj, i)[2] * spin_collection_op(Spin_obj, i)[1] for i in 1:N]).data * sol.u[j]))
end

prop_gamma_instant = prop_emiss #./ prop_Pop_ee
print(steps)
excited = [prop_gamma_instant[i] for i in 1:steps]
times = LinRange(0, tf, steps)
Plots.plot(times, excited, label="", xlabel="Time", ylabel="Photo emission rate", linewidth=2)


N = 3
tf = 3.0
dt = 0.001
H = Matrix(Spin_system_object.H_eff.data)
linds = [[AVQD.TagOperator(L[i], "i", 2)] for i in 1:N]
Hv = VectorizedEffectiveHamiltonian([(t) -> 1.0], [H], γ, linds)
u0 = zeros(2^N)
u0[1] = 1.0 + 0im
# u0=u0 |> normalize
ansatz = Ansatz(u0, relrcut=1e-1, vectorize=true, pool="all2")
sol2 = heresolve_avq(Hv, ansatz, [0, tf], dt, "Dicke_superradiance.csv")

jldsave("dicke_vectorized_0.1.jld2"; sol2.u)

# steps = Integer(tf / dt)
# print(steps)
# excited=[Float64(sol.u[i][1]) for i in 1:steps]
# ground=[Float64(real(sol.u[i][16])) for i in 1:steps]

# times = LinRange(0, tf, steps)
# Plots.plot(times, excited, label="Excited State", xlabel="Time(s)", ylabel="Amplitude", linewidth=2)
# plot!(times, ground, label="Ground State", linewidth=2)


a = load("dicke_vectorized_0.1.jld2")
u_avqd = get(a, "u", "zero")
steps = 3000

N = 3

prop_emiss = zeros(Float64, steps)
prop_Pop_ee = zeros(Float64, steps)
for j in 1:steps
    prop_emiss[j] = real(tr(sum((Spin_obj.gamma_diag[i]) * (dagger(Spin_obj.J_diag[i]) * Spin_obj.J_diag[i]) for i = 1:N).data * u_avqd[j]))
    prop_Pop_ee[j] = real(tr(sum([spin_collection_op(Spin_obj, i)[2] * spin_collection_op(Spin_obj, i)[1] for i in 1:N]).data * u_avqd[j]))
end
avqd = [prop_emiss[i] for i in 1:steps]
times2 = LinRange(0, 3.0, 50)
avqd = avqd[1:60:end]
plot!(times2, avqd, seriestype=:scatter, mc=[1], ms=3, markershape=:circle, markerstrokewidth=0.0, ma=0.8, label=L"AVQD $( d = 0.1λ)$", linewidth=2)
