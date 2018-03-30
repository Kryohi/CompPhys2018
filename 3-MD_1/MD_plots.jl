
include("./Periodic_Gas.jl")

## Grafico PV
XX, EE, TT, PP = AbstractArray, AbstractArray, AbstractArray, AbstractArray
N = 108
ρ = 0.05:0.025:2.0
P = zeros(ρ)
V = N./ρ

# Use a small fstep for the PV plot, but higher (~50) to create the video
@time for i = 1:length(ρ)
    println("Run ", i, "/", length(ρ))
    XX, EE, TT, PP, CM = simulation(N=108, T0=0.5, rho=ρ[i], maxsteps=2*10^4, fstep=10, dt=2e-4, csv=false)
    P[i] = mean(PP[length(PP)÷4:end])  #+ ρ[i]*TT[length(PP)÷4:end])
    #pp = plot!(PP)
end

PV1 = plot(V, P, xaxis=("V",(0,2250)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
#PV1 = plot(ρ, P, xaxis=("ρ",(0,2.0)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
gui()
