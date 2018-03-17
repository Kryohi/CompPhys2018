
using Plots, LaTeXStrings
pyplot()

function boxMuller(sigma, N, x0=0.0)
    srand(42)   # sets the rng seed, to obtain reproducible numbers
    c = Array{Float64}(N)
    for j = 1:round(Int,N/2)
        x1, x2 = rand(2)
        c[j*2] = sqrt(-2sigma*log(1-x1))*cos(2π*x2)
        c[j*2-1] = sqrt(-2sigma*log(1-x2))*cos(2π*x1)
    end
    return c
end

function shiftSystem!(A,L)
    for j in eachindex(A)
        @inbounds A[j] = A[j] - L*round(A[j]/L)
    end
end

function initializeSystem(N::Int, density, T)
    L = cbrt(N/density)
    σ = T*0.042      # T * k_b /m in qualche unità di misura

    Na = round(Int,∛(N/4)) # numero celle per dimensione
    a = L / Na  # passo reticolare
    if Na - ∛(N/4) != 0
        error("Can't make a cubic FCC crystal with this number of particles :(")
    end

    X = Array{Float64}(3N)
    for i=0:Na-1, j=0:Na-1, k = 0:Na-1  # loop over every cell of the cfc lattice
        n = i*Na*Na + j*Na + k # unique number for each triplet i,j,k
        X[n*4+1], X[N+n*4+1], X[2N+n*4+1] = a*i, a*j, a*k # vertice celle
        X[n*4+2], X[N+n*4+2], X[2N+n*4+2] = a*i + a/2, a*j + a/2, a*k
        X[n*4+3], X[N+n*4+3], X[2N+n*4+3] = a*i + a/2, a*j, a*k + a/2
        X[n*4+4], X[N+n*4+4], X[2N+n*4+4] = a*i, a*j + a/2, a*k + a/2
    end
    X += 1e-9   # needed to avoid particles exactly at the edges of the box
    shiftSystem!(X,L)
    V = boxMuller(σ,3N)  # da sistemare per array
    return [X, V]
end

NN = 256    #number of particles
X, V = initializeSystem(NN, 256, 1)
scatter(X[1:NN], X[NN+1:2NN], X[2NN+1:3NN], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-0.5,0.5)), yaxis=("y",(-0.5,0.5)), zaxis=("z",(-0.5,0.5)))
gui()
