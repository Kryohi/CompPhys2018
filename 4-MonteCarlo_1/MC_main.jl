
if nprocs()<4
  addprocs(4)   # add local worker processes (where N is the number of logical cores)
end

using Plots, DataFrames, ProgressMeter, CSV, PyCall
push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/MC_sim.jl"))
@everywhere import MC  # add module with all the functions in Perodic_Gas.jl

@time XX, jj, je = MC.oscillators(N=1000, D=3.5, T0=10.0, maxsteps=2030)
