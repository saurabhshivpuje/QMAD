
# Create a 5x5 matrix of zeros
H = zeros(ComplexF64,5, 5)

# Modify specific elements
H[2, 2] = 0.0267
H[2, 3] = -0.0129
H[2, 4] = 0.000632
H[3, 2] = -0.0129
H[3, 3] = 0.0273
H[3, 4] = 0.00404
H[4, 2] = 0.000632
H[4, 3] = 0.00404
H[4, 4] = 0

H=(2*π*H/4.135e-15)

# # Display the resulting matrix
# println(H)
α=2*3e12
β=  2*5e8
γ=2*6.28e12
# σ=sqrt(α)
l1=zeros(ComplexF64,5,5)
l1[2,2]=sqrt(α)
l1op = [AVQD.TagOperator(l1, "1", 5)]

l2=zeros(ComplexF64,5,5)
l2[3,3]=sqrt(α)
l2op = [AVQD.TagOperator(l2, "2", 5)]

l3=zeros(ComplexF64,5,5)
l3[4,4]=sqrt(α)
l3op = [AVQD.TagOperator(l3, "3", 5)]

l4=zeros(ComplexF64,5,5)
l4[1,2]=sqrt(β)
l4op = [AVQD.TagOperator(l4, "4", 5)]

l5=zeros(ComplexF64,5,5)
l5[1,3]=sqrt(β)
l5op = [AVQD.TagOperator(l5, "5", 5)]

l6=zeros(ComplexF64,5,5)
l6[1,4]=sqrt(β)
l6op = [AVQD.TagOperator(l6, "6", 5)]

l7=zeros(ComplexF64,5,5)
l7[5,4]=sqrt(γ)
l7op = [AVQD.TagOperator(l7, "7", 5)]
typeof(l1)
# l7_array=l7[:]


tf=300e-15
dt=1e-15
# tf=450
# dt=10
u0=zeros(ComplexF64,5,5)
u0[2,2]=1
He = Hamiltonian([(s) -> 1.0], [H], unit=:ħ)
# He.size
# l=l1+l2+l3+l4+l5+l6+l7

ops=[l1,l2,l3,l4,l5,l6,l7]

gamma=1
linds=[Lindblad(gamma, o) for o in ops]
# print(typeof(linds))
linds=InteractionSet(linds...)
annealing =  Annealing(He, u0, interactions=linds)
sol = solve_lindblad(annealing, tf, alg=Tsit5(), abstol=1e-6, reltol=1e-6, saveat=0:dt:tf)
steps=300 #Integer(tf/dt)
# real(sol.u[30][7])
zero_e=[real(sol.u[i][1]) for i in 1:steps]
excited_e=[real(sol.u[i][7]) for i in 1:steps]
ground_e=[real(sol.u[i][13]) for i in 1:steps]
one_e=[real(sol.u[i][19]) for i in 1:steps]
two_e=[real(sol.u[i][25]) for i in 1:steps]
times = LinRange(0, 300, steps)

plot(times, excited_e, legend=:topright,label=false, xlabel="Time(fs)", color=[1], ylabel="Population", linewidth=1, grid=false, thickness_scaling=1.2, framestyle=:box)
# plot!(legend=:outerbottom, legendcolumns=2)
plot!(times, ground_e, label=false )
plot!(times, one_e, label=false)
plot!(times, two_e, label=false)
plot!(times, zero_e, label=false)

times = LinRange(0, 300, 150)
# for t in times:
one=load("fmo_vectorized.jld2", "one")
two = load("fmo_vectorized.jld2", "two")
zero = load("fmo_vectorized.jld2", "zero")
ground = load("fmo_vectorized.jld2", "ground")
excited = load("fmo_vectorized.jld2", "excited")
# end
print(typeof(excited))
# Get alternate points
one= one[1:2:end]
two = two[1:2:end]
zero= zero[1:2:end]
ground = ground[1:2:end]
excited = excited[1:2:end]

plot!(dpi=300)
plot!(times, excited, label=L"Site 1 $|1⟩$", seriestype=:scatter, mc=[1], ms=2, markershape=:circle, markerstrokewidth=0.0, ma=0.8)
plot!(times, ground, label=L"Site 2 $|2⟩$", seriestype=:scatter, mc=[2], ms=2, markershape=:circle, markerstrokewidth=0.0, ma=0.8)
plot!(times, one, label=L"Site 3 $|3⟩$", seriestype=:scatter, mc=[3], ms=2, markershape=:circle, markerstrokewidth=0.0, ma=0.8)
plot!(times, zero, label=L"Ground $|0⟩$", seriestype=:scatter, mc=[5], ms=2, markershape=:circle, markerstrokewidth=0.0, ma=0.8)
plot!(times, two, label=L"Sink $|4⟩$", seriestype=:scatter, mc=[4], ms=2, markershape=:circle, markerstrokewidth=0.0, ma=0.8)  


# savefig("fmo_complex.pdf")

# Create a 5x5 matrix of zeros
n=8
H = zeros(ComplexF64,n, n)

# Modify specific elements
H[2, 2] = 0.0267
H[2, 3] = -0.0129
H[2, 4] = 0.000632
H[3, 2] = -0.0129
H[3, 3] = 0.0273
H[3, 4] = 0.00404
H[4, 2] = 0.000632
H[4, 3] = 0.00404
H[4, 4] = 0

H=(2*π*H/4.135e-15)

# # Display the resulting matrix
# println(H)
α=2*3e12
β=2*5e8
γ=2*6.28e12

# α=2*3e-3 #* 1e15
# β=0 #2*5e-7 #*1e15
# γ=0 #2*6.28e-3 #*1e15
# σ=sqrt(α)
l1=zeros(ComplexF64,n,n)
l1[2,2]=sqrt(α)
l1op = [AVQD.TagOperator(l1, "1", 6)]

l2=zeros(ComplexF64,n,n)
l2[3,3]=sqrt(α)
l2op = [AVQD.TagOperator(l2, "2", 6)]

l3=zeros(ComplexF64,n,n)
l3[4,4]=sqrt(α)
l3op = [AVQD.TagOperator(l3, "3", 6)]

l4=zeros(ComplexF64,n,n)
l4[1,2]=sqrt(β)
l4op = [AVQD.TagOperator(l4, "4", 6)]

l5=zeros(ComplexF64,n,n)
l5[1,3]=sqrt(β)
l5op = [AVQD.TagOperator(l5, "5", 6)]

l6=zeros(ComplexF64,n,n)
l6[1,4]=sqrt(β)
l6op = [AVQD.TagOperator(l6, "6", 6)]

l7=zeros(ComplexF64,n,n)
l7[5,4]=sqrt(γ)
l7op = [AVQD.TagOperator(l7, "7", 6)]
# l7_array=l7[:]

tf=300e-15
dt=1e-15
# linds=[l1op,l2op,l3op,l4op,l5op,l6op,l7op]
linds=[l1op,l2op,l3op,l4op,l5op,l6op,l7op]
# gamma=[0,0,0]#,0,0,0,0]
gamma=[1,1,1,1,1,1,1]
Hv = VectorizedEffectiveHamiltonian([(t) -> 1.0], [H], gamma, linds)
# # Hv_padded = vcat(Hv, zeros(7, 25))
# # Hv_padded = hcat(Hv_padded, zeros(32, 7))
# # println(Hv)
# # u0=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] |> normalize
# # u0 = ones(ComplexF64, 2^5) |> normalize
u0=[0,1,0,0,0,0,0,0] |> normalize
# typeof(u0)
# # println(length(u0))
# # u0=zeros(1,32)
# # u0[1,1]=1
# # print(u0)
ansatz = Ansatz(u0, relrcut=1e-3 , vectorize=true, pool="all2")
# # get_state(A::Ansatz) = A.state


# # ψ = ansatz |> get_state
# # println(ansatz.partial_theta)
# # println(length(A.A))
# # println(ansatz)
# # println()
sol2 = heresolve_avq(Hv, ansatz, [0, tf], dt,"FMO_complex.csv")

steps=300 #Integer(tf/dt)
# real(sol.u[30][7])
zero=[real(sol2.u[i][1]) for i in 1:steps]
excited=[real(sol2.u[i][10]) for i in 1:steps]
ground=[real(sol2.u[i][19]) for i in 1:steps]
one=[real(sol2.u[i][28]) for i in 1:steps]
two=[real(sol2.u[i][37]) for i in 1:steps]
# times = LinRange(0, tf, steps)
plot!(times, excited, seriestype=:scatter, mc=[2], ms=3, ma=0.8)
plot!(times, ground,seriestype=:scatter, mc=[2], ms=3, ma=0.8)
plot!(times, one, seriestype=:scatter, mc=[2], ms=3, ma=0.8)
plot!(times, two, seriestype=:scatter, mc=[2], ms=3, ma=0.8)
plot!(times, zero, seriestype=:scatter, mc=[2], ms=3, ma=0.8)