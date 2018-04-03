
if nprocs()<4
  addprocs(4)
end

@everywhere include("./Periodic_Gas.jl")
@everywhere import Sim

pyplot(size = (800, 600))
PyPlot.PyObject(PyPlot.axes3D)  # servirà finché non esce la prossima versione di Plots con bug fixato
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))

@everywhere function avgAtEquilibrium(A, f=3)   # where f is the fraction of steps to cut off
    l = length(A)
    return mean(A[l÷f:end]), std(A[l÷f:end])/sqrt(l*(1-1/f))
end

##
## Grafico PV
##
@everywhere function parallelPV(rho)
    N = 108
    T0 = 1.5
    V = N./rho
    # Use a small fstep (even 1) for the PV plot, but higher (20-50) to create the animation
    println("Run ", rho, "/", 2.4)
    XX, EE, TT, PP, = Sim.simulation(N=N, T0=T0, rho=rho, maxsteps=1*10^5, fstep=4, dt=5e-4, anim=false, csv=false, onlyP=true)
    P, dP = avgAtEquilibrium(PP)  #+ ρ[i]*TT[length(PP)÷4:end])
    E, dE = avgAtEquilibrium(EE)
    T, dT = avgAtEquilibrium(TT)
    return P, dP, E, dE, T, dT
    #make2DtemporalPlot(XX[:,1:2000], T=T0, rho=ρ[i], save=true)
end

ρ = 0.05:0.1:2.4
N = 108
T0 = 1.5
V = N./ρ
@time result = pmap(parallelPV, ρ)
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]

DP = convert(DataFrame, [V P])
file = string("./3-MD_1/Data/PV_",N,"_T",T0,".csv")
CSV.write(file, DP)

PV1 = plot(V, P, ribbon=dP, fillalpha=.3, xaxis=("V",(0,2250)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
savefig(PV1,"PV108_0.5_01to2.pdf")
#PV1 = plot(ρ, P, xaxis=("ρ",(0,2.0)), yaxis=("P",(0,5)),  linewidth=2, leg=false)
gui()
