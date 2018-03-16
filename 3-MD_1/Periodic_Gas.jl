
using Plots, LaTeXStrings
pyplot()

function boxMuller(sigma, N, x0=0) # sarebbe meglio sfruttare entrambi i numeri disponibili
    @show x1, x2 = rand(2)
    @show c1 = sqrt(2sigma*log(1-x1)*cos(2π*x2))
    #c2 = sqrt(2sigma*log(1-x2)*cos(2π*x1))
    return exp(-(c1 - x0)/2sigma^2)/sqrt(2π*sigma^2)
end

function shiftSystem(A,L) # sarebbe meglio se modificasse direttamente il vettore che gli si dà in pasto (funzioni con ! davanti)
    A .- L.*round.(A./L)
end

function initializeSystem(N::Int, density, T)
    L = (N/density)^(1/3)
    σ = T*4.2 #da sistemare
    #X = rand(3N).*L - L/2 # posizioni random dentro la scatola
    X = Array{Float64}(3N)
    Na = round(Int,(N/4)^(1/3)) # numero celle per dimensione
    a = L / Na  # passo reticolare
    for i=0:Na-1, j=0:Na-1, k = 0:Na-1  # loop over every cell of the cfc lattice
        n = i*Na*Na + j*Na + k # unique number for each triplet i,j,k
        X[n*4+1], X[N+n*4+1], X[2N+n*4+1] = a*i, a*j, a*k # vertice celle
        X[n*4+2], X[N+n*4+2], X[2N+n*4+2] = a*i + a/2, a*j + a/2, a*k
        X[n*4+3], X[N+n*4+3], X[2N+n*4+3] = a*i + a/2, a*j, a*k + a/2
        X[n*4+4], X[N+n*4+4], X[2N+n*4+4] = a*i, a*j + a/2, a*k + a/2
    end
    X = shiftSystem(X+1e-9,L) # +1e-9 needed to avoid particles exactly at the edges of the box
    V = Array{Float64}(3N)
    V = boxMuller(σ,N)  # da sistemare per array
    return [X, V]
end

XX, VV = initializeSystem(32, 32, 1)
scatter(VV[1:32], VV[33:64], VV[65:96],m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-0.5,0.5)), yaxis=("y",(-0.5,0.5)), zaxis=("z",(-0.5,0.5)))
gui()
