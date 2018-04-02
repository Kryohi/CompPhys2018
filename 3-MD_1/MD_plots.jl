
include("./Periodic_Gas.jl")
pyplot(size = (800, 600))

function averageAtEquilibrium(A, f=3)   # where f is the fraction of steps to cut off
    l = length(A)
    return mean(A[l÷f:end]), std(A[l÷f:end])/sqrt(l*(1-1/f))
end

## Grafico PV
XX, EE, TT, PP = AbstractArray, AbstractArray, AbstractArray, AbstractArray
N = 108
ρ = 0.1:0.1:2.0
P, dP = zeros(ρ), zeros(ρ)
E, dE = zeros(ρ), zeros(ρ)
T, dT = zeros(ρ), zeros(ρ)
T0 = 0.5
V = N./ρ

# Use a small fstep (even 1) for the PV plot, but higher (20-50) to create the video
@time for i = 1:length(ρ)
    println("Run ", i, "/", length(ρ))
    XX, EE, TT, PP, = simulation(N=N, T0=T0, rho=ρ[i], maxsteps=15*10^3, fstep=30, dt=2e-4, anim=false, csv=true)
    P[i], dP[i] = averageAtEquilibrium(PP)  #+ ρ[i]*TT[length(PP)÷4:end])
    E[i], dE[i] = averageAtEquilibrium(EE)
    T[i], dT[i] = averageAtEquilibrium(TT)
    make2DtemporalPlot(XX, T=T0, rho=ρ[i], save=true)
end

DP = convert(DataFrame, [V P])
file = string("./3-MD_1/Data/PV_",N,"_T",T0,".csv")
CSV.write(file, DP)

PV1 = plot(V, P, ribbon=dP, fillalpha=.3, xaxis=("V",(0,2250)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
savefig(PV1,"PV256_0.5_005to2.pdf")
#PV1 = plot(ρ, P, xaxis=("ρ",(0,2.0)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
gui()
