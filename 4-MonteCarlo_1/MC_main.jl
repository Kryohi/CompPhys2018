
using Statistics, Distributed, Plots, DataFrames, CSV, ProgressMeter
push!(LOAD_PATH, pwd())
include(string(pwd(), "/MC_sim.jl"))

nprocs()<4 && addprocs(4)   # add local worker processes (N is the num of logical cores)
@everywhere push!(LOAD_PATH, pwd()) # add current working directory to LOAD path
@everywhere include(string(pwd(), "/MC_sim.jl"))
@everywhere using Statistics, FFTW, Distributed

# Df is the initial Δ step value (as a fraction of a) and should be chosen quite carefully,
# even if it gets optimized during the burn-in
# some good values are ~1/70 for 32 particles and ~1/100 for 108, but it also depends on T and ρ
#@time EE, PP, jj, C_H, CV, CV2, = MC.metropolis_ST(N=108, T=0.4, rho=0.14, maxsteps=20*10^6, Df=1/100)


##
## Simulazioni multiple
##

@everywhere function parallelMC(rho, N, T, Tarray)
    @info string("Run ", findfirst(Tarray.==T), "/", length(Tarray))
    # Df iniziale andrebbe ottimizzato anche per T
    EE, PP, jj, C_H, CV, CV2, OP = MC.metropolis_ST(N=N, T=T, rho=rho, maxsteps=28*10^6, Df=(1/70), bmaxs=36*10^5) # 1/80 or less for ρ very small

    τ = sum(C_H)
    E, dE = mean(EE), std(EE[1:ceil(Int,τ/5):end])   # usare variance2?
    P, dP = mean(PP), std(PP[1:ceil(Int,τ/1000):end])
    OP, dOP = mean(OP), std(OP)
    @info string("Run ", findfirst(Tarray.==T),
    " finished, with tau = ", τ, " (T = ", T, ", rho = ", rho, ")")

    # Reweighting
    #T2 = [T-0.00667; T+0.00667] # delta in modo da far uscire punti equispaziati
    T2 = [T-0.003333; T+0.0033333]
    if T<0.7
        @info string("Reweighting distribution at ", round(T2[1]*1000)/1000, " and ", round(T2[2]*1000)/1000)
        Pr = MC.simpleReweight(T, T2, PP, EE[1:200:end])    # 200 sarebbe l'fstep deprecato...
        Er = MC.simpleReweight(T, T2, EE, EE)
        @time EEr1 = MC.energyReweight(T, T2[1], EE)
        EEr2 = MC.energyReweight(T, T2[2], EE)
        # To replace with
        #CVr, CVr2 = Array{Union{Float64,Missing}}(missing,2), Array{Union{Float64,Missing}}(missing,2)
        # when Plots will be missings-compliant
        CVr, CVr2 = ones(2).*NaN, ones(2).*NaN
        if length(EEr1) > 5*10^5 && length(EEr2) > 5*10^5
            C_H1, C_H2 = MC.fft_acf(EEr1, 42000), MC.fft_acf(EEr2, 42000)
            τ1, τ2 = sum(C_H1), sum(C_H2)
            CVr[1] = MC.cv(EEr1, T2[1], C_H1)
            CVr2[1] = MC.variance(EEr1[1:ceil(Int,τ1/5):end])/T2[1]^2 + 1.5T2[1]
            CVr[2] = MC.cv(EEr2, T2[2], C_H2)
            CVr2[2] = MC.variance(EEr2[1:ceil(Int,τ2/5):end])/T2[2]^2 + 1.5T2[2]
        end
        @show reweight_data = [T2 Er Pr CVr CVr2]
    else
        reweight_data = zeros(length(T2),5)
    end
    return P, dP, E, dE, CV, CV2, τ, OP, reweight_data
end

T = [0.04:0.02:0.16; 0.18:0.01:0.72; 0.76:0.04:1.28] # set per lavoro tutta notte # aumentare divisore se ρ bassa
#T = [0.04:0.02:0.7; 0.72:0.04:1.26]
N = 32
ρ = 0.21
V = N./ρ

# map the parallelPV function to the ρ array
@showprogress result = pmap(T0 -> parallelMC(ρ, N, T0, T), T)

# extract the resulting arrays from the result tuple
P, dP = [ x[1] for x in result ], [ x[2] for x in result ]
E, dE = [ x[3] for x in result ], [ x[4] for x in result ]
CV, CV2 = [ x[5] for x in result ], [ x[6] for x in result ]
τ = [ x[7] for x in result ]    # utile solo temporaneamente
OP = [ x[8] for x in result ]
reweight_data = [ x[9] for x in result ]
rd = filter(x -> x ≠ zeros(size(reweight_data[1])), reweight_data)
rd = vcat(rd...) # see splatting in docs


# save and plot the data
data = DataFrame(T=T, E=E, P=P, Cv=CV, Cv2=CV2, tau=τ, OP=OP, dE=dE, dP=dP)
file = string("./Data/MC_",N,"_rho",ρ,"_T",T[1],"-",T[end],".csv")
CSV.write(file, data)
data_r = DataFrame(T=rd[:,1], E=rd[:,2], P=rd[:,3], Cv=rd[:,4], Cv2=rd[:,5])
file = string("./Data/MC_reweighted_",N,"_rho",ρ,"_T",T[1],"-",T[end],".csv")
CSV.write(file, data_r)

P1 = plot(T, CV2, label="CV2", reuse = false)
gui()
file = string("./Plots/Tcv_",N,"_rho",ρ,"_T",T[1],"-",T[end],".pdf")
savefig(P1,file)
P2 = scatter(T, CV, label="CV", reuse = false)
scatter!(data_r[:T], data_r[:Cv], label="CV reweight")  # doesn't filter zeros yet
gui()
file = string("./Plots/TCv_",N,"_rho",ρ,"_T",T[1],"-",T[end],".pdf")
savefig(P2,file)
