
if nprocs()<4
  addprocs(4)
end


@everywhere include("/home/kryohi/Uni/CompPhys2018/3-MD_1/Periodic_Gas.jl")
push!(LOAD_PATH, "/home/kryohi/Uni/CompPhys2018/3-MD_1")
#@everywhere reload("Sim")
@everywhere import Sim

@everywhere using Plots, DataFrames
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
@everywhere function parallelPV(rho, N, T0)
    #N = 108
    #T0 = 0.5
    # Use a small fstep (even 1) for the PV plot, but higher (20-50) to create the animation
    println("Run ", rho, "/", 2.4)
    XX, EE, TT, PP, = Sim.simulation(N=N, T0=T0, rho=rho, maxsteps=6*10^4, fstep=30, dt=5e-4, anim=false, csv=false, onlyP=false)
    P, dP = avgAtEquilibrium(PP)  #+ ρ[i]*TT[length(PP)÷4:end])
    E, dE = avgAtEquilibrium(EE)
    T, dT = avgAtEquilibrium(TT)
    Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    return P, dP, E, dE, T, dT
end

ρ = [0.05:0.025:1.0; 1.05:0.05:1.95; 2.0:0.1:3.5]
ρ = 0.1:0.1:3.0
N = 108
T0 = 3.0
V = N./ρ
@time result = pmap(rho -> parallelPV(rho, N, T0), ρ)
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
T, dT = [ x[5] for x in result ], [ x[6] for x in result ]

DP = convert(DataFrame, [ρ V P])
file = string("./3-MD_1/Data/PV_",N,"_T",T0,".csv")
#CSV.write(file, DP)

rV1 = plot(ρ, P, xaxis=("ρ",(0,ρ[end])), yaxis=("P",(0,ceil(P[end]))), linewidth=2, leg=false)
file = string("./3-MD_1/Plots/rV_",N,"_T",T0,".pdf")
#savefig(rV1,file)

PV1 = plot(V, P, ribbon=dP, fillalpha=.3, xaxis=("V",(0,2500)), yaxis=("P",(0,ceil(P[end]))), linewidth=2, leg=false)
file = string("./3-MD_1/Plots/PV_",N,"_T",T0,".pdf")
#savefig(PV1,file)

gui()


## prove varie
#XX, EE, TT, PP, = Sim.simulation(N=108, T0=0.5, rho=2.1, maxsteps=1*10^5, fstep=4, dt=5e-4, anim=false)
