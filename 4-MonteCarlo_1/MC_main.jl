
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using Plots, DataFrames, ProgressMeter, CSV
push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/MC_sim.jl"))
@everywhere import MC  # add module with all the functions in MC_sim.jl

#@time XX, jj, je = HO.oscillators(N=1000, D=3.5, T0=10.0, maxsteps=2030)

@time XX, EE, PP, CV, je, jj = MC.metropolis_ST(N=500, T=2.0, rho=0.1, maxsteps=4000,
 fstep=2, Df=1/50)

@time E, P, CV, je, jj = MC.metropolis_MP(N=500, T=2.0, rho=0.1, maxsteps=2000, Df=1/50)
