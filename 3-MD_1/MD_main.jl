
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using DataFrames, CSV, ProgressMeter, PyCall, Plots, StaticArrays
push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/Periodic_Gas.jl"))
@everywhere import Sim  # add module with all the functions in Periodic_Gas.jl

pyplot(size=(800, 600))
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))


@everywhere function parallelPV(rho, N, T0, rhoarray)
    println("Run ", find(rhoarray.==rho)[1], "/", length(rhoarray))

    XX, CM, EE, TT, PP1, PP2 = Sim.simulation(N=N, T0=T0, rho=rho, maxsteps=10*10^4,
     fstep=20, dt=5e-4, anim=false, csv=false, onlyP=false)

    E, dE = Sim.avgAtEquilibrium(EE)
    T, dT = Sim.avgAtEquilibrium(TT)
    P1, dP1 = Sim.avgAtEquilibrium(PP1)
    P2, dP2 = Sim.avgAtEquilibrium(PP2)
    P, dP = P1+P2, sqrt(dP1^2 + dP2^2)
    op = Sim.orderParameter(XX, rho)
    #Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    return P, dP, E, dE, T, dT, op, P1, dP1, P2, dP2
end

##
## Grafico PV
##

ρ = 0.075:0.025:1.15
N = 256
T0 = 15.0
V = N./ρ

# map the parallelPV function to the ρ array
@time result = pmap(rho -> parallelPV(rho, N, T0, ρ), ρ)
# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
T, dT = [ x[5] for x in result ], [ x[6] for x in result ]
op = [ x[7] for x in result ]
P1, dP1 = [ x[8] for x in result ], [ x[9] for x in result ]
P2, dP2 = [ x[10] for x in result ], [ x[11] for x in result ]


data = DataFrame(d=ρ, V=V, P=P, dP=dP, E=E, dE=dE, T=T, dT=dT,
 op=op, P1=P1, dP1=dP1, P2=P2, dP2=dP2)
file = string("./Data/PV2_",N,"_T",T0,".csv")
CSV.write(file, data)

rV1 = Plots.plot(ρ, P, ribbon=dP, fillalpha=.3, xaxis=("ρ",(0,ceil(ρ[end]*10)/10)),
 yaxis=("P",(0,ceil(P[end]))), linewidth=2, leg=false)
file = string("./Plots/rV2_",N,"_T",T0,".pdf")
savefig(rV1,file)

PV1 = Plots.plot(V, P, ribbon=dP, fillalpha=.3, xaxis=("V",(0,2000)), yaxis=("P",(-1,ceil(P[end]))), linewidth=2, leg=false)

file = string("./Plots/PV2_",N,"_T",T0,".pdf")
savefig(PV1,file)

gui()


## prove varie

#XX, CM, EE, TT, PP1, PP2 = Sim.simulation(N=256, T0=0.025, rho=0.01, maxsteps=20*10^4,
# fstep=80, dt=5e-4, anim=true, csv=true)
# @show Sim.avg3D(CM)
# Sim.make2DtemporalPlot(XX[:,100:200], T=1.0, rho=0.4, save=true)
# Sim.make3Dplot(CM, T=1.0, rho=1.3)
# ld = Sim.lindemann2(XX, CM, 108, 1.1)
#OP = Sim.orderParameter(XX, 0.05)
#PPP = Plots.plot([1:20:length(PP1)*20].*5e-4, PP1, xaxis=("t"), yaxis=("P"), linewidth=1.5, leg=false)
#Plots.plot!(PPP,[1:20:length(PP2)*20].*5e-4, PP2, xaxis=("t"), yaxis=("P"), linewidth=1.5, leg=false)
gui()
#Plots.plot([1:20:length(TT)*20].*5e-4, TT, xaxis=("t"), yaxis=("P"), linewidth=1.5, leg=false)
