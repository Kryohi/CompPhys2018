using Statistics, FFTW, Distributed, ProgressMeter, PyCall, Plots

HO(x::Array{Float64},k::Float64) = k.*x.^2 ./2
force(x::Array{Float64},k::Float64) = -k.*x

# creates an array with length N of gaussian distributed numbers using Box-Muller
function vecboxMuller(sigma::Float64, N::Int, x0=0.0)
    #srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end

# toy model of canonical ensemble of harmonic oscillators
function oscillator(; N=500, T=0.5, Δ=1e-2, γ=0.5, maxsteps=10^4)

    A = γ*T
    σ = sqrt(2A*Δ)/γ
    k = .1
    X = vecboxMuller(k*10,3N)    # inizializzazione come gaussiana, ma non necessario
    X = rand(3N)*2 .+ 1.0
    Y = zeros(3N)
    ap = zeros(3N)
    j = zeros(Int64, maxsteps)
    #X, jeq = equilibriumOSC(X,D,T0)

    for n=1:maxsteps
        # Proposta
        gauss = vecboxMuller(σ,3N)
        Y .= X .+ Δ.*force(X,k) .+ gauss.*σ     #force da dividere per γ?
        WX = (X.-Y.-force(X,k).*Δ).^2   # controllare segni
        WY = (Y.-X.-force(Y,k).*Δ).^2
        ap = exp.((HO(X,k).-HO(Y,k))./T .+ (WX.-WY)./(4*Δ*T))
        η = rand(3N)
        for i = 1:length(X)
            if η[i] < ap[i]
                X[i] = Y[i]
                j[n] += 1
            end
        end
    end
    return X, j./(3N)
end
