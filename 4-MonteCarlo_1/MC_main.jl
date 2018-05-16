
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using Plots, DataFrames, ProgressMeter, CSV
@everywhere push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/MC_sim.jl"))
@everywhere import MC  # add module with all the functions in MC_sim.jl

#@time XX, jj, je = HO.oscillators(N=1000, D=3.5, T0=10.0, maxsteps=2030)

@time XX, EE, PP, CV, je, jj, C_H, CV, CV2 = MC.metropolis_ST(N=108, T=2.0,
 rho=0.1, maxsteps=15000, fstep=1, Df=1/60)

@time E, P, CV, je, jj = MC.metropolis_MP(N=500, T=2.0, rho=0.1, maxsteps=2000, Df=1/50)


##
## Simulazioni multiple
##

@everywhere function parallelPV(rho, N, T, Tarray)
    println("Run ", find(Tarray.==T)[1], "/", length(Tarray))

    XX, EE, PP, CV, je, jj, C_H, CV, CV2 = MC.metropolis_ST(N=N, T=T, rho=rho, maxsteps=20000,
     fstep=1, Df=1/60)

    E, dE = mean(EE), std(EE)
    P, dP = mean(PP), std(PP)
    #op = Sim.orderParameter(XX, rho)
    #Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    return P, dP, E, dE, CV, CV2
end

T = [0.01:0.01:0.15; 0.2:0.1:0.82]
N = 108
ρ = 0.6
V = N./ρ

#@time for T0 in Tarr
# map the parallelPV function to the ρ array
@time result = pmap(T0 -> parallelPV(ρ, N, T0, T), T)
# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
CV = [ x[5] for x in result ]
CVignorante = [ x[6] for x in result ]

plot(T,CVignorante)
gui()
