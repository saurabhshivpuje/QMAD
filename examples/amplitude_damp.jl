using 
tf=1000e-12 
dt = 1000e-14 
u0 = [1/4 0; 0 3/4]
# u0=[1/1 0;0 0]
H=Matrix{ComplexF64}(I,2,2)
H = Hamiltonian([(s) -> 1.0], [H], unit=:ħ)
ops=Complex[0.0 1; 0.0 0.0]
print(typeof(ops))
gamma=1.52e9
linds=(Lindblad(gamma, ops))
print(typeof(Lindblad))

linds=InteractionSet(linds)

annealing =  Annealing(H, u0, interactions=linds)
sol = solve_lindblad(annealing, tf, alg=Tsit5(), abstol=1e-6, reltol=1e-6, saveat=0:dt:tf)
# σp = (σx + 1im * σy) 
# σm = (σx - 1im * σy) 
# # sol.u[1]
# tr(σp*σm*sol.u[1])
excited_e=[Float64(sol.u[i][4]) for i in 1:100]
ground_e=[Float64(sol.u[i][1]) for i in 1:100]
times = LinRange(0, 1000, 100)
plot(times, excited_e, label=L"State $|1⟩$ (exact)", xlabel="Time(ps)", color=[1], ylabel="Population", linewidth=1.5, grid=false, legend=:right, thickness_scaling=1.2, framestyle=:box)
plot!(times, ground_e, label=L"State $|0⟩$ (exact)", color=[2], linewidth=1.5)
plot!(dpi=300)

# AVQD Vectorization results

H=Matrix{ComplexF64}(I,2,2)
gamma = 1.52e9 # * 1e-12
tf = 1000e-12 #* 1e12
dt = 4000e-14 #* 1e12
steps= 25 #Integer(tf/dt)
σp = (σx + 1im * σy)/2

linds = [AVQD.TagOperator(single_clause([σp], [i], 1.0, 1), "σ₊"*string(i), 1) for i in 1:1]
H = VectorizedEffectiveHamiltonian([(t) -> 1.0], [H], gamma, linds)

u0=[1/2,sqrt(3)/2] |> normalize
ansatz = Ansatz(u0, relrcut=1e-6, vectorize=true, pool="all2")
res = solve_avq(H, ansatz, [0, tf], dt)
excited=[Float64(res.u[i][4]) for i in 1:steps]
ground=[Float64(res.u[i][1]) for i in 1:steps]
times = LinRange(0, 1000, steps)
plot!(times, excited, label=L"State $|1⟩$ (UAVQD)", mc=[1], seriestype=:scatter, ms=3.5, markerstrokewidth=0.0, ma=0.8)
plot!(times, ground, label=L"State $|0⟩$ (UAVQD)", seriestype=:scatter, mc=[2], ms=3.5, markerstrokewidth=0.0, ma=0.8)
savefig("amplitude_damping.pdf")

