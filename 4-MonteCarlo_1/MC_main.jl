
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

# Df is the initial Δ step value (as a fraction of a) and should be chosen quite carefully,
# even if it gets optimized during the burn-in
# some good values are ~1/40 for 32 particles and ~1/70 for 128, but it also depends on T and ρ
#@time EE, PP, jj, C_H, CV, CV2, = MC.metropolis_ST(N=32, T=0.5, rho=0.35, maxsteps=6*10^6, Df=1/42)


##
## Simulazioni multiple
##


@everywhere function parallelPV(rho, N, T, Tarray)
    info("Run ", find(Tarray.==T)[1], "/", length(Tarray))
    # Df iniziale andrebbe ottimizzato anche per T
    EE, PP, jj, C_H, CV, CV2, = MC.metropolis_ST(N=N, T=T, rho=rho, maxsteps=10*10^6, Df=(1/45))

    info("Run ", find(Tarray.==T)[1], " finished, with tau = ", sum(C_H))
    E, dE = mean(EE), std(EE)   # usare variance2?
    P, dP = mean(PP), std(PP)
    τ = sum(C_H)

    return P, dP, E, dE, CV, CV2, τ
end

T = [0.06:0.02:0.56; 0.6:0.04:1.12] # set per lavoro tutta notte
#T = 0.06:0.02:1.16
N = 32
ρ = 0.28
V = N./ρ

# map the parallelPV function to the ρ array
@time result = pmap(T0 -> parallelPV(ρ, N, T0, T), T)

# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
CV, CVignorante = [ x[5] for x in result ], [ x[6] for x in result ]
τ = [ x[7] for x in result ]

data = DataFrame(T=T, E=E, dE=dE, P=P, dP=dP, Cv=CV, Cv2=CVignorante, tau=τ)
file = string("./Data/MC_",N,"_rho",ρ,"_T",T[1],"-",T[end],".csv")
CSV.write(file, data)

P1 = plot(T, CVignorante, label="CV2", reuse = false)
gui()
file = string("./Plots/Tcv_",N,"_rho",ρ,"_T",T[1],"-",T[end],".pdf")
savefig(P1,file)
P2 = plot(T, CV, label="CV", reuse = false)
gui()
file = string("./Plots/TCv_",N,"_rho",ρ,"_T",T[1],"-",T[end],".pdf")
savefig(P2,file)
