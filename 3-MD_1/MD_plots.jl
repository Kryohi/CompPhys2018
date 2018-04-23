
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using Plots, DataFrames, ProgressMeter, CSV, PyCall
push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/Periodic_Gas.jl"))
@everywhere import Sim  # add module with all the functions in Perodic_Gas.jl

pyplot(size=(800, 600))
PyPlot.PyObject(PyPlot.axes3D)  # servirà finché non esce la prossima versione di Plots con bug fixato
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))

@everywhere function avgAtEquilibrium(A, f=4)   # where f is the fraction of steps to cut off
    l = length(A)
    return mean(A[l÷f:end]), std(A[l÷f:end])/sqrt(l*(1-1/f))
end

@everywhere function parallelPV(rho, N, T0, rhoarray)
    # Use a small fstep (even 1) for the PV plot, but higher (20-50) to create the animation
    println("Run ", find(rhoarray.==rho), "/", length(rhoarray))
    XX, EE, TT, PP, CM = Sim.simulation(N=N, T0=T0, rho=rho, maxsteps=1*10^5, fstep=10, dt=5e-4,
     anim=false, csv=false, onlyP=false)
    P, dP = avgAtEquilibrium(PP)  #+ ρ[i]*TT[length(PP)÷4:end])
    E, dE = avgAtEquilibrium(EE)
    T, dT = avgAtEquilibrium(TT)
    #ld = Sim.lindemann2(XX, CM, N, rho)
    op = Sim.orderParameter(XX, N, rho)
    #Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    return P, dP, E, dE, T, dT, op
end

##
## Grafico PV
##

ρ = 0.075:0.025:1.125
N = 108
T0 = 7.0
V = N./ρ

# map the parallelPV function to the ρ array
@time result = pmap(rho -> parallelPV(rho, N, T0, ρ), ρ)
# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
T, dT = [ x[5] for x in result ], [ x[6] for x in result ]
op = [ x[7] for x in result ]


DP = convert(DataFrame, [ρ V P dP E dE T dT op])
rename!(DP, f => t for (f, t) = zip([:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9],
 [:d :V :P :dP :E :dE :T :dT :op]))
file = string("./Data/PV_",N,"_T",T0,".csv")
CSV.write(file, DP)

rV1 = plot(ρ, P, ribbon=dP, fillalpha=.3, xaxis=("ρ",(0,ρ[end])), yaxis=("P",(0,ceil(P[end]))), linewidth=2, leg=false)
file = string("./Plots/rV_",N,"_T",T0,".pdf")
savefig(rV1,file)

PV1 = plot(V, P, ribbon=dP, fillalpha=.3, xaxis=("V",(0,2000)), yaxis=("P",(0,ceil(P[end]))), linewidth=2, leg=false)
file = string("./Plots/PV_",N,"_T",T0,".pdf")
savefig(PV1,file)

gui()


## prove varie
#XX, EE, TT, PP, CM = Sim.simulation(N=108, T0=1.0, rho=0.9, maxsteps=5*10^4, fstep=10, dt=5e-4, anim=false, csv=true)
# @show Sim.avg3D(CM)
# Sim.make2DtemporalPlot(XX[:,100:200], T=1.0, rho=0.4, save=true)
# Sim.make3Dplot(CM, T=1.0, rho=1.3)
# ld = Sim.lindemann2(XX, CM, 108, 1.1)
#OP = Sim.orderParameter(XX, 108, 0.1)
