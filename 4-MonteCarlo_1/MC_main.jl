
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using Plots, DataFrames, ProgressMeter, CSV
@everywhere using Plots, DataFrames, ProgressMeter, CSV
push!(LOAD_PATH, pwd())
include(string(pwd(), "/MC_sim.jl"))
import MC
@everywhere push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/MC_sim.jl"))
@everywhere import MC  # add module with all the functions in MC_sim.jl


#@time XX, EE, PP, jj, C_H, CV, CV2 = MC.metropolis_ST(N=108, T=0.22,
# rho=0.3, maxsteps=200000, fstep=1, Df=1/70)

##
## Simulazioni multiple
##

@everywhere function saveCSV(rho, N, T, EE, PP, CV, CV2, C_H)
    data = DataFrame(E=EE, P=PP, CVcorr=CV, CV=CV2, Ch=[C_H; missings(length(EE)-length(C_H))])
    file = string("./Data/MCtemp_",N,"_rho",rho,"_T",T,".csv")
    CSV.write(file, data)
    info("Data saved in ", file)
end

@everywhere function parallelPV(rho, N, T, Tarray)
    info("Run ", find(Tarray.==T)[1], "/", length(Tarray))

    XX, EE, PP, jj, C_H, CV, CV2 = MC.metropolis_ST(N=N, T=T, rho=rho, maxsteps=400000, fstep=1, Df=(1/70)*N/108)

    saveCSV(rho, N, T, EE, PP, CV, CV2, C_H)
    E, dE = mean(EE), std(EE)
    P, dP = mean(PP), std(PP)
    #op = Sim.orderParameter(XX, rho)
    #Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    return P, dP, E, dE, CV, CV2
end

T = [0.1:0.025:0.6; 0.65:0.5:1.5]
#T = 0.2:0.1:1.4
N = 108
ρ = 0.4
V = N./ρ

# map the parallelPV function to the ρ array
@time result = pmap(T0 -> parallelPV(ρ, N, T0, T), T)

# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
CV = [ x[5] for x in result ]
CVignorante = [ x[6] for x in result ]

data = DataFrame(T=T, E=E, dE=dE, P=P, dP=dP, Cv=CV, Cv2=CVignorante)
file = string("./Data/MC_",N,"_rho",ρ,".csv")
CSV.write(file, data)

P1 = plot(T,CVignorante)
gui()
file = string("./Plots/Tcv_",N,"_rho",ρ,"_T",T[1],"-",T[end],".pdf")
savefig(P1,file)
